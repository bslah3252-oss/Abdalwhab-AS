[app]

# Application title
title = Abdalwhab-AS

# Package name
package.name = abdalwhab-as

# Package domain (reverse-DNS)
package.domain = org.dsd

# Source directory (where main.py lives)
source.dir = .

# Source files to include
source.include_exts = py,png,jpg,kv,atlas,task

# Additional files to bundle (the Mediapipe model)
source.include_patterns = face_landmarker.task

# Application version
version = 1.0.0

# Application requirements
# Note: mediapipe on Android requires the --headless opencv variant
requirements = python3==3.11.0,kivy==2.3.0,numpy,pillow,opencv-python-headless,mediapipe

# Application icon (optional — place a 512x512 PNG at this path)
# icon.filename = %(source.dir)s/data/icon.png

# Orientation (portrait or landscape)
orientation = portrait

# Supported Android ABIs
android.archs = arm64-v8a, armeabi-v7a

# Android API target
android.api = 33

# Minimum Android SDK version
android.minapi = 24

# Android NDK version (r25b is stable for most native libs)
android.ndk = 25b

# Android SDK version
android.sdk = 33

# Use gradle build (required for modern Android)
android.gradle_dependencies = "androidx.core:core-ktx:1.10.1"

# Accept Android SDK licenses automatically
android.accept_sdk_license = True

# Android permissions
android.permissions = INTERNET, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, CAMERA, READ_MEDIA_IMAGES

# Entry point
p4a.branch = master

# Python-for-Android bootstrap
android.bootstrap = sdl2

# Keep app in foreground
android.wakelock = False

# Fullscreen mode (0 = windowed with status bar, 1 = fullscreen)
fullscreen = 0

# Android release keystore (fill in for production builds)
# android.keystore = my-release-key.jks
# android.keystore_alias = my-key-alias
# android.keystore_passwd = mypassword
# android.keyalias_passwd = mypassword

# Logcat filters for debugging
android.logcat_filters = *:S python:D

# Copy libraries into the APK
android.copy_libs = 1

# Enable backup
android.allow_backup = True

[buildozer]

# Log level (0 = error, 1 = info, 2 = debug)
log_level = 2

# Warn if buildozer is run as root
warn_on_root = 1

# Directory for build artifacts
build_dir = ./.buildozer

# Directory for the built APK
bin_dir = ./bin
