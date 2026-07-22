"""
augment_rusts.py
ساخت نسخه‌های واقعاً جدید (نه کپی) از عکس‌های حاوی rusts با Albumentations
برخلاف oversampling ساده، اینجا هر نسخه واقعاً پیکسل‌های متفاوتی داره
"""

import os
import cv2
import albumentations as A

# ---------- تنظیمات ----------
TRAIN_IMAGES_DIR = "/content/drive/MyDrive/casting_defect_project/train/images"
TRAIN_LABELS_DIR = "/content/drive/MyDrive/casting_defect_project/train/labels"
RUSTS_CLASS_ID = 2              # ⚠️ با data.yaml چک کن
NUM_AUGMENTED_COPIES = 2        # هر عکس rusts، ۲ نسخه‌ی جدید (متفاوت) از خودش می‌سازه

# ---------- تعریف augmentation ----------
# نکته‌ی مهم: چون rust شکل لکه‌مانند/بدون‌جهت غالبه (برخلاف scratches که خطیه)،
# چرخش اینجا کم‌خطرتره - ولی محتاط بمون و شدید نزن (درس فولاد رو یادت باشه)
transform = A.Compose(
    [
        A.Rotate(limit=25, p=0.7),                          # چرخش ملایم (rust خطی نیست، ریسک کمتر از scratches)
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.4),          # شبیه‌سازی نویز دوربین صنعتی
        A.HorizontalFlip(p=0.5),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)


def read_yolo_labels(label_path):
    """خواندن باکس‌ها و کلاس‌ها از فایل لیبل YOLO"""
    boxes, class_labels = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            class_labels.append(int(parts[0]))
            boxes.append([float(p) for p in parts[1:5]])
    return boxes, class_labels


def write_yolo_labels(label_path, boxes, class_labels):
    """نوشتن باکس‌های تغییریافته به فایل لیبل جدید"""
    with open(label_path, "w") as f:
        for cls, box in zip(class_labels, boxes):
            f.write(f"{cls} {' '.join(map(str, box))}\n")


def has_rusts(label_path):
    if not os.path.exists(label_path):
        return False
    with open(label_path) as f:
        return any(int(line.split()[0]) == RUSTS_CLASS_ID for line in f)


# ---------- پردازش ----------
count_augmented = 0

for filename in os.listdir(TRAIN_IMAGES_DIR):
    label_name = filename.rsplit(".", 1)[0] + ".txt"
    label_path = os.path.join(TRAIN_LABELS_DIR, label_name)

    if not has_rusts(label_path):
        continue                                              # فقط عکس‌های rusts رو پردازش کن

    image = cv2.imread(os.path.join(TRAIN_IMAGES_DIR, filename))
    boxes, class_labels = read_yolo_labels(label_path)

    for i in range(NUM_AUGMENTED_COPIES):
        try:
            result = transform(image=image, bboxes=boxes, class_labels=class_labels)
        except Exception as e:
            print(f"⚠️ خطا روی {filename}: {e}")
            continue

        base_name = filename.rsplit(".", 1)[0]
        ext = filename.rsplit(".", 1)[1]
        new_img_name = f"{base_name}_aug{i}.{ext}"
        new_label_name = f"{base_name}_aug{i}.txt"

        cv2.imwrite(os.path.join(TRAIN_IMAGES_DIR, new_img_name), result["image"])
        write_yolo_labels(
            os.path.join(TRAIN_LABELS_DIR, new_label_name),
            result["bboxes"],
            result["class_labels"],
        )

    count_augmented += 1

print(f"✅ {count_augmented} عکس rusts پردازش شد، هر کدام {NUM_AUGMENTED_COPIES} نسخه‌ی جدید و متفاوت گرفت.")
print("حالا count_classes.py رو دوباره اجرا کن تا تعداد جدید rusts رو ببینی.")


"""
============================================================
خلاصه‌ی کد:

هدف کد:
    ساخت نسخه‌های واقعاً جدید (نه کپی عین) از عکس‌های حاوی rusts، با
    استفاده از Albumentations، تا مدل الگوهای متنوع‌تری از این کلاس
    کم‌نمونه ببینه - برخلاف oversampling ساده که فقط تکرار عین بود.

پیش‌نیازها:
    - pip install albumentations opencv-python
    - RUSTS_CLASS_ID درست تنظیم شده باشه

مراحل منطق کد:
    ۱. برای هر عکس train، چک می‌کنه آیا کلاس rusts داره
    ۲. اگه داشت، باکس‌ها و کلاس‌هاش رو از فایل لیبل می‌خونه
    ۳. Albumentations هم عکس هم باکس‌ها رو همزمان و هماهنگ تغییر میده
       (این نکته‌ی مهمیه: باکس‌ها با چرخش/فلیپ عکس خودشون هم جابه‌جا میشن،
       نه اینکه فقط عکس عوض بشه و باکس‌ها جای غلط بمونن)
    ۴. هر نسخه با اسم جدید (_aug0, _aug1) ذخیره میشه
    ۵. نتیجه: تعداد نمونه‌ی rusts چند برابر میشه، ولی هر نسخه واقعاً
       تصویر متفاوتیه (نه کپی بی‌فایده مثل قبل)

⚠️ نکته‌ی طراحی مهم: چرخش (Rotate) اینجا امنه چون rust شکل لکه‌مانند و
بدون جهت غالب داره - برخلاف crazing/scratches که خطی هستن و چرخش
بهشون آسیب می‌زد (طبق تجربه‌ی فولاد). اگه یک روز خواستی همین روش رو
برای spots یا scratches هم امتحان کنی، حتما این تصمیم رو دوباره بر
اساس شکل هندسی اون کلاس بررسی کن، کورکورانه کپی نکن.

این کد رو کامل به AI بسپار؛ تصمیمی که باید خودت بگیری فقط انتخاب
نوع/شدت augmentation بر اساس شکل هندسی کلاسه (که بالا توضیح دادم)،
نه جزئیات syntax کتابخونه.
============================================================
"""
