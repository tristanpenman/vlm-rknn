#include <jni.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <android/log.h>
#include <opencv2/imgcodecs.hpp>

#include "rknn_utils.h"
#include "vlm_rknn.h"

#define LOG_TAG "vlm-rknn-jni"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::mutex g_mutex;
std::unique_ptr<vlm_rknn::Session> g_session;
std::vector<float> g_imageEmbedding;
bool g_hasImage = false;

struct JniCallbackContext {
    JavaVM* vm = nullptr;
    jobject callback = nullptr;
    jmethodID onText = nullptr;
    jmethodID onFinish = nullptr;
    jmethodID onError = nullptr;
};

JNIEnv* acquireEnv(JavaVM* vm, bool* outAttached)
{
    JNIEnv* env = nullptr;
    jint rc = vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (rc == JNI_OK) {
        *outAttached = false;
        return env;
    }
    if (rc == JNI_EDETACHED && vm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
        *outAttached = true;
        return env;
    }
    return nullptr;
}

void releaseEnv(JavaVM* vm, bool attached)
{
    if (attached) {
        vm->DetachCurrentThread();
    }
}

bool resolveCallbackMethods(JNIEnv* env, jobject callback, JniCallbackContext* ctx)
{
    jclass cls = env->GetObjectClass(callback);
    if (cls == nullptr) {
        LOGE("could not get callback class");
        return false;
    }
    ctx->onText = env->GetMethodID(cls, "onText", "(Ljava/lang/String;)V");
    ctx->onFinish = env->GetMethodID(cls, "onFinish", "()V");
    ctx->onError = env->GetMethodID(cls, "onError", "()V");
    env->DeleteLocalRef(cls);
    if (ctx->onText == nullptr || ctx->onFinish == nullptr || ctx->onError == nullptr) {
        LOGE("could not resolve RknnLlmCallback methods");
        return false;
    }
    return true;
}

void dispatchCallback(const char* text, LLMCallState state, JniCallbackContext* ctx)
{
    if (ctx == nullptr || ctx->callback == nullptr) {
        return;
    }

    bool attached = false;
    JNIEnv* env = acquireEnv(ctx->vm, &attached);
    if (env == nullptr) {
        LOGE("could not acquire JNIEnv in callback");
        return;
    }

    if (state == RKLLM_RUN_NORMAL) {
        if (text != nullptr) {
            jstring jtext = env->NewStringUTF(text);
            if (jtext != nullptr) {
                env->CallVoidMethod(ctx->callback, ctx->onText, jtext);
                env->DeleteLocalRef(jtext);
            }
        }
    } else if (state == RKLLM_RUN_FINISH) {
        env->CallVoidMethod(ctx->callback, ctx->onFinish);
    } else if (state == RKLLM_RUN_ERROR) {
        env->CallVoidMethod(ctx->callback, ctx->onError);
    }

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    releaseEnv(ctx->vm, attached);
}

jint rknnLlmInit(JNIEnv* env, jclass, jstring jmodelFamily, jstring jencoderPath, jstring jllmPath)
{
    if (jmodelFamily == nullptr || jencoderPath == nullptr || jllmPath == nullptr) {
        LOGE("model path is null");
        return -1;
    }

    const char* modelFamily = env->GetStringUTFChars(jmodelFamily, nullptr);
    if (modelFamily == nullptr) {
        return -1;
    }
    const char* encoderPath = env->GetStringUTFChars(jencoderPath, nullptr);
    if (encoderPath == nullptr) {
        env->ReleaseStringUTFChars(jmodelFamily, modelFamily);
        return -1;
    }
    const char* llmPath = env->GetStringUTFChars(jllmPath, nullptr);
    if (llmPath == nullptr) {
        env->ReleaseStringUTFChars(jencoderPath, encoderPath);
        env->ReleaseStringUTFChars(jmodelFamily, modelFamily);
        return -1;
    }

    vlm_rknn::ModelConfig config;
    if (!vlm_rknn::parseModelFamily(modelFamily, config.modelFamily)) {
        LOGE("unknown model family: %s", modelFamily);
        env->ReleaseStringUTFChars(jllmPath, llmPath);
        env->ReleaseStringUTFChars(jencoderPath, encoderPath);
        env->ReleaseStringUTFChars(jmodelFamily, modelFamily);
        return -1;
    }
    if (vlm_rknn::modelFamilyUsesVisionEncoder(config.modelFamily)) {
        config.visionEncoderPath = encoderPath;
    }
    config.languageModelPath = llmPath;
    config.maxNewTokens = 512;
    config.maxContextLen = 4096;
    config.numCores = 3;

    auto session = std::make_unique<vlm_rknn::Session>(std::move(config));
    jint ret = static_cast<jint>(session->init());

    env->ReleaseStringUTFChars(jllmPath, llmPath);
    env->ReleaseStringUTFChars(jencoderPath, encoderPath);
    env->ReleaseStringUTFChars(jmodelFamily, modelFamily);

    std::lock_guard<std::mutex> lock(g_mutex);
    if (ret == 0) {
        g_session = std::move(session);
        g_imageEmbedding.clear();
        g_hasImage = false;
    }
    return ret;
}

jint rknnLlmSetImage(JNIEnv* env, jclass, jstring jimagePath)
{
    if (jimagePath == nullptr) {
        LOGE("image path is null");
        return -1;
    }

    const char* imagePath = env->GetStringUTFChars(jimagePath, nullptr);
    if (imagePath == nullptr) {
        return -1;
    }

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    env->ReleaseStringUTFChars(jimagePath, imagePath);
    if (image.empty()) {
        LOGE("could not decode image");
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_session || !g_session->isReady()) {
        LOGE("not initialised");
        return -1;
    }
    if (!vlm_rknn::modelFamilyUsesVisionEncoder(g_session->config().modelFamily)) {
        LOGE("active model family does not use a vision encoder");
        return -1;
    }

    const auto& encoder = g_session->visionEncoder();
    cv::Mat preprocessed;
    jint ret = static_cast<jint>(vlm_rknn::preprocessImageForVisionEncoder(
        g_session->config().modelFamily,
        image,
        cv::Size(encoder.modelWidth, encoder.modelHeight),
        preprocessed));
    if (ret != 0) {
        LOGE("image preprocessing failed: %d", static_cast<int>(ret));
        return ret;
    }

    const std::size_t total = static_cast<std::size_t>(encoder.modelImageToken)
        * static_cast<std::size_t>(encoder.modelEmbedSize)
        * static_cast<std::size_t>(encoder.ioNum.n_output);
    g_imageEmbedding.assign(total, 0.0f);

    ret = static_cast<jint>(g_session->encode(preprocessed.data, g_imageEmbedding.data()));

    if (ret != 0) {
        g_imageEmbedding.clear();
        g_hasImage = false;
        LOGE("image encoding failed: %s", rknn_utils::rknnErrorMessage(ret));
        return ret;
    }

    g_hasImage = true;
    return 0;
}

jint rknnLlmRun(JNIEnv* env, jclass, jstring jprompt, jobject jcallback)
{
    if (jprompt == nullptr) {
        LOGE("prompt is null");
        return -1;
    }

    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);
    if (prompt == nullptr) {
        return -1;
    }

    JniCallbackContext ctx;
    if (env->GetJavaVM(&ctx.vm) != JNI_OK) {
        env->ReleaseStringUTFChars(jprompt, prompt);
        return -1;
    }

    if (jcallback != nullptr) {
        if (!resolveCallbackMethods(env, jcallback, &ctx)) {
            env->ReleaseStringUTFChars(jprompt, prompt);
            return -1;
        }
        ctx.callback = env->NewGlobalRef(jcallback);
    }

    jint ret = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_session || !g_session->isReady()) {
            LOGE("not initialised");
            ret = -1;
        } else if (g_session->promptContainsImage(prompt) && !g_hasImage) {
            LOGE("prompt references an image but no image is loaded");
            ret = -1;
        } else {
            g_session->setOutputCallback([&ctx](const char* text, LLMCallState state) {
                dispatchCallback(text, state, &ctx);
            });
            ret = static_cast<jint>(g_session->decode(prompt, g_hasImage ? g_imageEmbedding.data() : nullptr));
            if (ret != 0) {
                LOGE("decoder run failed: %d", static_cast<int>(ret));
            }
            g_session->setOutputCallback(nullptr);
        }
    }

    if (ctx.callback != nullptr) {
        env->DeleteGlobalRef(ctx.callback);
    }
    env->ReleaseStringUTFChars(jprompt, prompt);
    return ret;
}

void rknnLlmCleanup(JNIEnv*, jclass)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_session.reset();
    g_imageEmbedding.clear();
    g_hasImage = false;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*)
{
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        LOGE("could not retrieve JNIEnv");
        return JNI_ERR;
    }

    jclass cls = env->FindClass("com/tristanpenman/vlmrknn/RknnLlm");
    if (cls == nullptr) {
        LOGE("could not find RknnLlm class");
        return JNI_ERR;
    }

    const JNINativeMethod methods[] = {
        {
            "nativeInit",
            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)I",
            reinterpret_cast<void*>(rknnLlmInit),
        },
        {"nativeSetImage", "(Ljava/lang/String;)I", reinterpret_cast<void*>(rknnLlmSetImage)},
        {
            "nativeRun",
            "(Ljava/lang/String;Lcom/tristanpenman/vlmrknn/RknnLlmCallback;)I",
            reinterpret_cast<void*>(rknnLlmRun),
        },
        {"nativeCleanup", "()V", reinterpret_cast<void*>(rknnLlmCleanup)},
    };

    if (env->RegisterNatives(cls, methods, sizeof(methods) / sizeof(methods[0])) != JNI_OK) {
        LOGE("could not register native methods");
        return JNI_ERR;
    }

    return JNI_VERSION_1_6;
}
