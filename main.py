
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# مسیر پوشه documents
documents_folder = Path(__file__).parent / "documents"


# خواندن فایل متنی
def read_txt(file_path):
    return file_path.read_text(encoding="utf-8")


# خواندن فایل PDF
def read_pdf(file_path):
    reader = PdfReader(str(file_path))
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages_text.append(
                f"\n[صفحه {page_number}]\n{text}"
            )

    return "\n".join(pages_text)


# تقسیم متن به بخش‌های کوچک
def split_into_chunks(text, chunk_size=500, overlap=100):
    text = " ".join(text.split())

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size or overlap")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # تا جای ممکن متن را وسط یک کلمه قطع نکن
        if end < len(text):
            last_space = text.rfind(" ", start, end)

            if last_space > start + chunk_size // 2:
                end = last_space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


# خواندن فایل‌ها
documents = []

for file_path in documents_folder.iterdir():

    if not file_path.is_file():
        continue

    if file_path.suffix.lower() == ".txt":
        content = read_txt(file_path)

    elif file_path.suffix.lower() == ".pdf":
        content = read_pdf(file_path)

    else:
        continue

    documents.append({
        "source": file_path.name,
        "text": content
    })


# تقسیم متن فایل‌ها به بخش‌های کوچک
all_chunks = []

for document in documents:
    chunks = split_into_chunks(document["text"])

    for chunk_index, chunk_text in enumerate(chunks, start=1):
        all_chunks.append({
            "source": document["source"],
            "chunk_id": chunk_index,
            "text": chunk_text
        })

# بررسی وجود متن برای جست‌وجو
if not all_chunks:
    print("هیچ متن قابل جست‌وجویی پیدا نشد.")
    raise SystemExit


# 1. بارگذاری مدل Embedding
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)


# 2. تبدیل متن بخش‌ها به بردارهای عددی
chunk_texts = [
    "passage: " + chunk["text"]
    for chunk in all_chunks
]

chunk_embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_tensor=True,
    normalize_embeddings=True
)


# 3. بارگذاری مدل تولید پاسخ
print("Loading language model...")

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=-1
)

print("\nچت‌بات آماده است!")
print("برای خروج، کلمه exit را بنویس.")


# 4. دریافت سؤال‌های کاربر
while True:
    question = input("\nسؤال تو: ").strip()

    if question.lower() in ("exit", "quit"):
        print("خدانگهدار!")
        break

    if not question:
        continue


    # 5. پیدا کردن بخش‌های مرتبط با سؤال
    query_embedding = embedding_model.encode(
        "query: " + question,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    scores = util.cos_sim(
        query_embedding,
        chunk_embeddings
    )[0]

    top_k = min(3, len(all_chunks))
    top_results = scores.topk(k=top_k)


    # 6. آماده‌کردن متن‌های بازیابی‌شده
    retrieved_chunks = []

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):
        chunk = all_chunks[int(index)]
        retrieved_chunks.append(chunk)

    context = "\n\n".join(
        f"[منبع: {chunk['source']} | "
        f"بخش: {chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )


    # 7. ساخت پیام برای مدل زبانی
    messages = [
        {
            "role": "system",
            "content": (
                "تو یک دستیار پرسش‌وپاسخ فارسی هستی. "
                "فقط بر اساس اطلاعات موجود در متن‌های ارائه‌شده "
                "پاسخ بده. اگر اطلاعات کافی نیست، صادقانه بگو "
                "که در فایل‌های بازیابی‌شده اطلاعات کافی "
                "برای پاسخ وجود ندارد. از خودت اطلاعات نساز. "
                "پاسخت روشن و مختصر باشد."
            )
        },
        {
            "role": "user",
            "content": (
                f"متن‌های بازیابی‌شده:\n{context}\n\n"
                f"سؤال کاربر: {question}\n\n"
                "بر اساس متن‌های بالا پاسخ بده."
            )
        }
    ]


    # 8. تولید پاسخ
    print("\nدر حال تولید پاسخ...")

    result = generator(
        messages,
        max_new_tokens=160,
        do_sample=False
    )

    answer = result[0]["generated_text"][-1]["content"]

    print("\nپاسخ چت‌بات:")
    print(answer)


    # 9. نمایش فایل‌های بازیابی‌شده
    sources = list(dict.fromkeys(
        chunk["source"] for chunk in retrieved_chunks
    ))

    print("\nفایل‌های بازیابی‌شده:")

    for source in sources:
        print(f"- {source}")



