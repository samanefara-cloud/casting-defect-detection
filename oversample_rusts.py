"""
oversample_rusts.py
تکرار فیزیکی عکس‌هایی که کلاس rusts دارن، برای متعادل کردن دیتاست
(فقط rusts - چون فرضیه‌ی کمبود نمونه فقط برای این کلاس تایید شد)
"""

import os
import shutil

# ---------- تنظیمات ----------
TRAIN_IMAGES_DIR = "/content/drive/MyDrive/casting_defect_project/train/images"
TRAIN_LABELS_DIR = "/content/drive/MyDrive/casting_defect_project/train/labels"
RUSTS_CLASS_ID = 2          # ⚠️ حتما با data.yaml چک کن این عدد درسته (ترتیب: scratches=0, spots=1, rusts=2)
EXTRA_COPIES = 1            # هر عکس rusts، ۱ بار اضافه تکرار میشه (یعنی در مجموع ۲ برابر دیده میشه)

def has_rusts(label_path):
    """چک می‌کنه آیا این فایل لیبل، حداقل یک باکس از کلاس rusts داره یا نه"""
    if not os.path.exists(label_path):
        return False
    with open(label_path) as f:
        for line in f:
            class_id = int(line.split()[0])
            if class_id == RUSTS_CLASS_ID:
                return True
    return False


# ---------- پیدا کردن و تکرار عکس‌های rusts ----------
count_duplicated = 0

for filename in os.listdir(TRAIN_IMAGES_DIR):
    label_name = filename.rsplit(".", 1)[0] + ".txt"
    label_path = os.path.join(TRAIN_LABELS_DIR, label_name)

    if has_rusts(label_path):
        base_name = filename.rsplit(".", 1)[0]
        ext = filename.rsplit(".", 1)[1]

        for copy_num in range(1, EXTRA_COPIES + 1):
            new_img_name = f"{base_name}_dup{copy_num}.{ext}"
            new_label_name = f"{base_name}_dup{copy_num}.txt"

            shutil.copy(
                os.path.join(TRAIN_IMAGES_DIR, filename),
                os.path.join(TRAIN_IMAGES_DIR, new_img_name)
            )
            shutil.copy(
                label_path,
                os.path.join(TRAIN_LABELS_DIR, new_label_name)
            )
        count_duplicated += 1

print(f"✅ {count_duplicated} عکس حاوی rusts پیدا و {EXTRA_COPIES} بار تکرار شد.")
print("حالا count_classes.py رو دوباره اجرا کن تا مطمئن بشی توازن بهتر شده.")


"""
============================================================
خلاصه‌ی کد:

هدف کد:
    تکرار فیزیکی عکس‌هایی که کلاس rusts دارن (کپی فایل عکس+لیبل با
    اسم جدید)، تا مدل توی هر epoch این کلاس رو بیشتر ببینه - بدون
    اینکه عکس جدید واقعی جمع‌آوری بشه.

پیش‌نیازها:
    - مسیر درست TRAIN_IMAGES_DIR و TRAIN_LABELS_DIR
    - RUSTS_CLASS_ID درست باشه (از data.yaml چک کن)

مراحل منطق کد:
    ۱. برای هر عکس توی پوشه‌ی train، فایل لیبل هم‌نامش رو می‌خونه
    ۲. اگه لیبل حاوی کلاس rusts بود، عکس+لیبل رو با اسم جدید (_dup1)
       کپی می‌کنه - این یعنی همون عکس، یک نسخه‌ی اضافه توی دیتاست داره
    ۳. چاپ تعداد عکس‌هایی که تکرار شدن

⚠️ نکته‌ی مهم: این کد را فقط روی rusts اجرا کن (مطابق فرضیه‌ای که
تایید کردیم)، نه روی spots - چون مشکل spots شباهت بصری با scratches
هست، نه کمبود تعداد، و oversampling این مشکل رو حل نمی‌کنه.

این کد رو کامل به AI بسپار؛ چیزی که باید خودت تصمیم بگیری، فقط
EXTRA_COPIES ـه (چند بار تکرار کافیه) - عدد ۱ (یعنی ۲ برابر شدن)
نقطه‌ی شروع محافظه‌کارانه‌ست؛ زیاد کردنش بیشتر، ریسک overfitting
روی همون عکس‌های تکراری رو بالا می‌بره.
============================================================
"""
