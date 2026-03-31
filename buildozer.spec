[app]
title = Smile Design Al-Nabil
package.name = smiledesign
package.domain = org.nabil
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1

# المكتبات المطلوبة بدقة لعمليات تحليل الصور والذكاء الاصطناعي
requirements = python3,kivy==2.3.0,kivymd,mediapipe,opencv-python-headless,numpy

orientation = portrait
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# إعدادات النظام المحدثة 2026
android.api = 34
android.minapi = 21
# android.sdk = 33
android.ndk = 26b
android.archs = arm64-v8a, armeabi-v7a
android.allow_backup = True

[buildozer]
log_level = 2
warn_on_root = 1
