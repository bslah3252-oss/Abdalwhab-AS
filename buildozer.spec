[app]
title = Smile Design Al-Nabil
package.name = smiledesign
package.domain = org.nabil
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1

# المتطلبات الأساسية للتطبيق
requirements = python3,kivy,mediapipe,opencv-python

orientation = portrait
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# الإعدادات المحدثة لضمان النجاح
android.api = 34
android.minapi = 21
# تم تعطيل السطر التالي ليتولى النظام تحديده تلقائياً وتجنب الخطأ السابق
# android.sdk = 33
android.ndk = 26b
android.archs = arm64-v8a, armeabi-v7a
android.allow_backup = True

[buildozer]
log_level = 2
warn_on_root = 1
