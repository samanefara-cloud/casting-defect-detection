"""
train_casting.py
آموزش مدل تشخیص عیب ریخته‌گری (۳ کلاس) با YOLOv8n
فریز شده: فقط backbone (لایه‌های ۰ تا ۹)
"""

import os                                    # 🔧 اضافه‌شده: برای os.path.exists, os.makedirs, os.path.join
import torch
from ultralytics import YOLO

# 1️⃣ تشخیص دستگاه (GPU یا CPU)
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"🔍 دستگاه مورد استفاده: {device}")

# مسیر data.yaml (متغیر سراسری که قبلاً توی نوت‌بوک تعریف کردی)
data_yaml_path = gdrive_data_yaml_path

# --- بررسی وجود فایل قبل از ترین (برای اطمینان) ---
print(f"Diagnostic: Checking if {data_yaml_path} exists...")
if os.path.exists(data_yaml_path):
    print(f"Diagnostic: {data_yaml_path} EXISTS.")
    print("Diagnostic: Content of data.yaml:")
    !cat {data_yaml_path}
else:
    print(f"Diagnostic: {data_yaml_path} DOES NOT EXIST. This is unexpected.")

# 2️⃣ بارگذاری مدل خام (وزن‌های آماده‌ی COCO)
model = YOLO('yolov8n.pt')

# ❄️ فریز کردن ۱۰ لایه‌ی اول (دقیقاً backbone، چون SPPF لایه‌ی ۹ ـه)
num_freeze_layers = 10
print(f"🥶 فریز کردن {num_freeze_layers} لایه اول مدل...")
for i in range(num_freeze_layers):
    if i < len(model.model.model):                    # جلوگیری از خطای index خارج از محدوده
        for param in model.model.model[i].parameters():
            param.requires_grad = False                # این لایه‌ها دیگه آپدیت نمیشن
    else:
        print(f"Warning: Model has fewer than {num_freeze_layers} layers. Frozen {i} layers.")
        break

# 3️⃣ ساخت پوشه‌ی ذخیره‌سازی دائمی روی گوگل درایو
gdrive_project_dir = os.path.join(gdrive_base_dir, 'runs', 'detect')
os.makedirs(gdrive_project_dir, exist_ok=True)
print(f"Ensured Google Drive project directory exists: {gdrive_project_dir}")

# 4️⃣ شروع آموزش
results = model.train(
    data=data_yaml_path,
    epochs=60,                        # 🔒 تنها تغییر نسبت به baseline (30→60)
    imgsz=640,
    batch=16,
    device=device,
    save_period=5,                    # هر ۵ اپاک یک چک‌پوینت ذخیره کن
    workers=4,
    resume=False,
    project=gdrive_project_dir,       # مستقیم روی Drive، نه /content
    name='real_CSDD_yolo_frozen_e60'  # اسم جدید، جدا از ران قبلی
)

print("✅ آموزش کامل شد!")


"""
============================================================
خلاصه‌ی کد:

هدف کد:
    آموزش مدل تشخیص عیب ریخته‌گری (۳ کلاس) با ترنسفر لرنینگ روی
    YOLOv8n، با فریز کردن backbone (لایه‌های ۰ تا ۹) - baseline
    امن مشابه چیزی که برای پروژه‌ی فولاد امتحان و تایید شد.

پیش‌نیازها:
    - ultralytics نصب باشه
    - gdrive_data_yaml_path و gdrive_base_dir از قبل توی نوت‌بوک تعریف
      شده باشن (متغیرهای سراسری)
    - Drive متصل (mount) شده باشه

مراحل منطق کد:
    ۱. تشخیص GPU/CPU
    ۲. چک کردن وجود فایل data.yaml (برای اطمینان قبل از شروع)
    ۳. بارگذاری مدل خام yolov8n
    ۴. فریز کردن ۱۰ لایه‌ی اول (backbone) - بر اساس تحلیل معماری
       (SPPF = مرز پایان backbone) + حجم دیتاست (۲۱۶۳ عکس، رده‌ی متوسط)
    ۵. ساخت پوشه‌ی دائمی روی Drive برای جلوگیری از گم‌شدن نتایج
    ۶. آموزش با epochs=30, imgsz=640 - دقیقاً baseline امتحان‌شده

نکته‌ی یادگیری کلیدی این پروژه: تصمیم freeze=10 از ترکیب دو چیز اومد -
(۱) دانستن دقیق مرز backbone در معماری ثابت YOLOv8n، و
(۲) حجم دیتاست خودت (نه یک عدد ثابت جهانی).
این بخش (تنظیم فلگ‌های train) رو می‌تونی به AI بسپاری؛ چیزی که باید
خودت مسلط باشی، همین منطق تصمیم‌گیریه، نه حفظ کردن syntax.
============================================================
"""
