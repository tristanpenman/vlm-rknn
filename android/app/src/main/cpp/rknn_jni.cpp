#include <jni.h>

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <android/log.h>

#include "qwen_vl_rknn.h"

#define LOG_TAG "qwen-vl-rknn-jni"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::mutex g_mutex;
std::unique_ptr<qwen_vl_rknn::Session> g_session;
std::vector<float> g_img_vec;
bool g_has_image = false;

struct JniCallbackContext {
    JavaVM* vm = nullptr;
    jobject callback = nullptr;
    jmethodID on_text = nullptr;
    jmethodID on_finish = nullptr;
    jmethodID on_error = nullptr;
};

JNIEnv* acquire_env(JavaVM* vm, bool* out_attached)
{
    JNIEnv* env = nullptr;
    jint rc = vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (rc == JNI_OK) {
        *out_attached = false;
        return env;
    }
    if (rc == JNI_EDETACHED && vm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
        *out_attached = true;
        return env;
    }
    return nullptr;
}

void release_env(JavaVM* vm, bool attached)
{
    if (attached) {
        vm->DetachCurrentThread();
    }
}

bool resolve_callback_methods(JNIEnv* env, jobject callback, JniCallbackContext* ctx)
{
    jclass cls = env->GetObjectClass(callback);
    if (cls == nullptr) {
        LOGE("could not get callback class");
        return false;
    }
    ctx->on_text = env->GetMethodID(cls, "onText", "(Ljava/lang/String;)V");
    ctx->on_finish = env->GetMethodID(cls, "onFinish", "()V");
    ctx->on_error = env->GetMethodID(cls, "onError", "()V");
    env->DeleteLocalRef(cls);
    if (ctx->on_text == nullptr || ctx->on_finish == nullptr || ctx->on_error == nullptr) {
        LOGE("could not resolve RknnLlmCallback methods");
        return false;
    }
    return true;
}

void dispatch_callback(const char* text, LLMCallState state, JniCallbackContext* ctx)
{
    if (ctx == nullptr || ctx->callback == nullptr) {
        return;
    }

    bool attached = false;
    JNIEnv* env = acquire_env(ctx->vm, &attached);
    if (env == nullptr) {
        LOGE("could not acquire JNIEnv in callback");
        return;
    }

    if (state == RKLLM_RUN_NORMAL) {
        if (text != nullptr) {
            jstring jtext = env->NewStringUTF(text);
            if (jtext != nullptr) {
                env->CallVoidMethod(ctx->callback, ctx->on_text, jtext);
                env->DeleteLocalRef(jtext);
            }
        }
    } else if (state == RKLLM_RUN_FINISH) {
        env->CallVoidMethod(ctx->callback, ctx->on_finish);
    } else if (state == RKLLM_RUN_ERROR) {
        env->CallVoidMethod(ctx->callback, ctx->on_error);
    }

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    release_env(ctx->vm, attached);
}

jint rknnLlm_init(JNIEnv* env, jclass, jstring jencoder_path, jstring jllm_path)
{
    if (jencoder_path == nullptr || jllm_path == nullptr) {
        LOGE("model path is null");
        return -1;
    }

    const char* encoder_path = env->GetStringUTFChars(jencoder_path, nullptr);
    if (encoder_path == nullptr) {
        return -1;
    }
    const char* llm_path = env->GetStringUTFChars(jllm_path, nullptr);
    if (llm_path == nullptr) {
        env->ReleaseStringUTFChars(jencoder_path, encoder_path);
        return -1;
    }

    qwen_vl_rknn::ModelConfig config;
    config.vision_encoder_path = encoder_path;
    config.language_model_path = llm_path;
    config.max_new_tokens = 512;
    config.max_context_len = 4096;
    config.num_cores = 3;

    auto session = std::make_unique<qwen_vl_rknn::Session>(std::move(config));
    jint ret = static_cast<jint>(session->init());

    env->ReleaseStringUTFChars(jllm_path, llm_path);
    env->ReleaseStringUTFChars(jencoder_path, encoder_path);

    std::lock_guard<std::mutex> lock(g_mutex);
    if (ret == 0) {
        g_session = std::move(session);
        g_img_vec.clear();
        g_has_image = false;
    }
    return ret;
}

jintArray rknnLlm_getImageInputSize(JNIEnv* env, jclass)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_session || !g_session->is_ready()) {
        return nullptr;
    }
    const auto& encoder = g_session->vision_encoder();
    jintArray arr = env->NewIntArray(2);
    if (arr == nullptr) {
        return nullptr;
    }
    jint values[2] = {encoder.model_width, encoder.model_height};
    env->SetIntArrayRegion(arr, 0, 2, values);
    return arr;
}

jint rknnLlm_setImage(JNIEnv* env, jclass, jbyteArray jpixels, jint width, jint height)
{
    if (jpixels == nullptr) {
        LOGE("pixels is null");
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_session || !g_session->is_ready()) {
        LOGE("not initialised");
        return -1;
    }

    const auto& encoder = g_session->vision_encoder();
    if (width != encoder.model_width || height != encoder.model_height) {
        LOGE("image dim mismatch: got %dx%d, want %dx%d",
             width, height, encoder.model_width, encoder.model_height);
        return -1;
    }

    const jsize len = env->GetArrayLength(jpixels);
    const jsize expected = width * height * 3;
    if (len != expected) {
        LOGE("pixel buffer size mismatch: got %d, want %d", static_cast<int>(len), static_cast<int>(expected));
        return -1;
    }

    const size_t total = static_cast<size_t>(encoder.model_image_token)
        * static_cast<size_t>(encoder.model_embed_size)
        * static_cast<size_t>(encoder.io_num.n_output);
    g_img_vec.assign(total, 0.0f);

    jbyte* pixels = env->GetByteArrayElements(jpixels, nullptr);
    if (pixels == nullptr) {
        g_img_vec.clear();
        return -1;
    }

    jint ret = static_cast<jint>(
        g_session->encode(reinterpret_cast<unsigned char*>(pixels), g_img_vec.data()));
    env->ReleaseByteArrayElements(jpixels, pixels, JNI_ABORT);

    if (ret != 0) {
        g_img_vec.clear();
        g_has_image = false;
        return ret;
    }

    g_has_image = true;
    return 0;
}

jint rknnLlm_run(JNIEnv* env, jclass, jstring jprompt, jobject jcallback)
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
        if (!resolve_callback_methods(env, jcallback, &ctx)) {
            env->ReleaseStringUTFChars(jprompt, prompt);
            return -1;
        }
        ctx.callback = env->NewGlobalRef(jcallback);
    }

    jint ret = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_session || !g_session->is_ready()) {
            LOGE("not initialised");
            ret = -1;
        } else if (g_session->prompt_contains_image(prompt) && !g_has_image) {
            LOGE("prompt references an image but no image is loaded");
            ret = -1;
        } else {
            g_session->set_output_callback([&ctx](const char* text, LLMCallState state) {
                dispatch_callback(text, state, &ctx);
            });
            ret = static_cast<jint>(g_session->decode(prompt, g_has_image ? g_img_vec.data() : nullptr));
            g_session->set_output_callback(nullptr);
        }
    }

    if (ctx.callback != nullptr) {
        env->DeleteGlobalRef(ctx.callback);
    }
    env->ReleaseStringUTFChars(jprompt, prompt);
    return ret;
}

void rknnLlm_cleanup(JNIEnv*, jclass)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_session.reset();
    g_img_vec.clear();
    g_has_image = false;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*)
{
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        LOGE("could not retrieve JNIEnv");
        return JNI_ERR;
    }

    jclass cls = env->FindClass("com/tristanpenman/qwenvlrknn/RknnLlm");
    if (cls == nullptr) {
        LOGE("could not find RknnLlm class");
        return JNI_ERR;
    }

    const JNINativeMethod methods[] = {
        {"nativeInit", "(Ljava/lang/String;Ljava/lang/String;)I", reinterpret_cast<void*>(rknnLlm_init)},
        {"nativeGetImageInputSize", "()[I", reinterpret_cast<void*>(rknnLlm_getImageInputSize)},
        {"nativeSetImage", "([BII)I", reinterpret_cast<void*>(rknnLlm_setImage)},
        {"nativeRun", "(Ljava/lang/String;Lcom/tristanpenman/qwenvlrknn/RknnLlmCallback;)I", reinterpret_cast<void*>(rknnLlm_run)},
        {"nativeCleanup", "()V", reinterpret_cast<void*>(rknnLlm_cleanup)},
    };

    if (env->RegisterNatives(cls, methods, sizeof(methods) / sizeof(methods[0])) != JNI_OK) {
        LOGE("could not register native methods");
        return JNI_ERR;
    }

    return JNI_VERSION_1_6;
}
