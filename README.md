Marketing Copywriter Fine-tuning: TinyLlama-1.1B on Alpaca Marketing Dataset

Тестовое задание: дообучить языковую модель под специфический стиль ответов.
Выбранный стиль - маркетинговый копирайтер: короткие, убедительные тексты с CTA, хэштегами и конкретикой.

---

Почему я выбрала маркетинговый стиль

Я могла взять что угодно: вопрос-ответ, обучающие диалоги, технические объяснения. Остановилась на маркетинговом копирайтинге, это конкретная профессиональная задача с измеримым результатом. Либо модель пишет как копирайтер, либо нет, не абстрактное "улучшить качество ответов". И первое я поставила себе 4 вопроса: что я хочу получить на выходе, есть ли под это данные, хватит ли ресурсов, и как я пойму что получилось. Что я хочу получить на выходе? Модель которая пишет в стиле маркетингового копирайтера коротко, убедительно, с CTA и конкретикой.
Есть ли данные под эту задачу? Не сразу. self_instruct не загрузился, dolly-15k оказался иллюзией - 516 "маркетинговых" записей на деле были статьями про Virgin Australia и рецептами. Нашла в alpaca-cleaned после итеративной фильтрации и ручной инспекции 380 записей.
Хватит ли ресурсов? TinyLlama-1.1B вместо Mistral-7B осознанный выбор под T4. QLoRA не запустилась из-за CUDA 13.0, перешла на LoRA в float16. Уложилась в бесплатный Colab.
Как я пойму что получилось? BERTScore для семантического сравнения с базовой моделью + ручной разбор 5 примеров где модели расходятся больше всего.

---

Шаг 1. Я разделила на несколько шагов и первым шагом был выбор датасета. Сначала сравнила три варианта:

| Датасет | Всего | Маркетинг | Средняя длина ответа |
|---|---|---|---|
| yizhongw/self_instruct | - | не загрузился (устаревший формат) | - |
| databricks/databricks-dolly-15k | 15 011 | 516 | 62 слова |
| yahma/alpaca-cleaned | 52 000 | 1 373 | 88 слов |

Насчет первого датасета yizhongw/self_instruct - создавался как разнообразный instruction датасет с реальными задачами, маркетинговых примеров там должно было быть много. Но при загрузке упал с ошибкой:

```
RuntimeError: Dataset scripts are no longer supported, but found self_instruct.py
```

HuggingFace с версии datasets 2.x перестал поддерживать датасеты на основе кастомных Python скриптов. self_instruct не был обновлён под новый формат (parquet), поэтому загрузить его не получилось. Перешла на databricks-dolly-15k как замену он современный и тоже написан людьми.

Почему отказалась от dolly-15k: просмотрела реальные примеры большинство из 516 "маркетинговых" записей попали туда случайно. Например, Virgin Australia (closed_qa) и Key Lime Pie (summarization) прошли фильтр потому что слово brand встречалось в тексте контекста. Реального маркетингового контента там почти не было.

В итоге выбрала alpaca-cleaned несмотря на синтетическое происхождение, маркетинговые примеры там качественнее и разнообразнее.

Следующий мой шаг это фильтрация. Я сделала в три этапа.

Сначала базовая очистка дубликатов: удаляла в два прохода. Сначала точные дубликаты по полю instruction убрала случаи когда одна и та же задача встречалась несколько раз с немного разными ответами. Потом нормализованные дубликаты убирала пунктуацию и пробелы, приводила к lowercase, и снова проверяла на совпадения. Это поймало пары типа "Write a slogan." и "Write a slogan" которые точный матч не видит.

1. Позитивный фильтр - ключевые слова в instruction: marketing, advertisement, slogan, tagline, campaign, copywriting, promotional, cta, product description, landing page, social media post...
2. Негативный фильтр - чтобы исключить нематркетинговые примеры. После первого прогона в датасете оказались: математика (perimeter, algorithm), личные письма (supervisor, pay raise), политика (election, democracy), D&D (dungeons and dragons). Проверяла вручную, на это потратила примерно 20 минут и добавила 60+ негативных паттернов.
3. Ручная инспекция - просмотрела ~380 записей, выявила паттерны которые автоматика пропустила. Пограничные случаи (come up with 5 use-cases, evaluate the effectiveness) убрала вручную.

Фильтрация качества:
- Минимальная длина output: 30 слов - ответы короче это обрывки, не маркетинговый текст
- Максимальная длина output: 600 слов - длиннее это уже статья, не копирайтинг
- Убрала ответы начинающиеся с "I cannot" и "I am unable" - отказы модели которые случайно попали в датасет
- Убрала записи с более чем 30 переносами строк - это списки и таблицы, не связный текст

Формат JSONL: каждая запись — одна строка, валидный JSON:
```json
{"instruction": "Write a tagline for a luxury skincare brand.", "response": "Age-defying beauty, naturally."}
```
Если у примера было поле input (контекст задачи) - склеивала его с instruction через \n\nInput:. Финальный файл проверяла построчным парсингом чтобы убедиться что нет битых записей.

---

После выбора датасета и фильтрации приступила к Fine-tuning

Выбор модели: изначально планировала Mistral-7B-Instruct-v0.2, но отказалась по двум причинам:
- На Colab T4 (15GB VRAM) при batch_size > 2 падал по памяти
- Время обучения 2+ часа на бесплатном Colab нестабильно

Выбрала TinyLlama-1.1B-Chat-v1.0 - быстро, стабильно.

Проблема с QLoRA: планировала QLoRA (4-bit через bitsandbytes), но столкнулась с проблемой совместимости: Colab использовал CUDA 13.0, а bitsandbytes на тот момент не имел скомпилированного бинарника под эту версию. Перепробовала версии 0.42.0, 0.43.3, 0.45.5 - все падали с libbitsandbytes_cuda130.so not found. Перешла на обычный LoRA в float16. Для TinyLlama-1.1B на T4 памяти хватает без квантизации (~2.2 GB модель).

LoRA конфигурация:
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    lora_dropout=0.05,
    task_type='CAUSAL_LM',
)
```

Почему r=16: начинала с r=8 loss падал слишком медленно. При r=32 модель переобучалась на val уже к концу 2-й эпохи. r=16 оказался оптимальным.

Почему все 4 модуля: сначала брала только q_proj и v_proj как в оригинальной статье LoRA. После добавления k_proj и o_proj loss стал сходиться стабильнее.

Параметры обучения:
```python
num_train_epochs=3
per_device_train_batch_size=4   # batch_size=8 давал OOM на T4
gradient_accumulation_steps=4   # эффективный batch = 16
learning_rate=2e-4
lr_scheduler_type='cosine'
warmup_ratio=0.03
max_seq_length=512
```

Loss curve:

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 1.2484 | 0.9709 |
| 2 | 0.9869 | 0.9534 |
| 3 | 0.9918 | 0.9516 |

Val loss стабилизировался к 3-й эпохе переобучения нет.

![Loss Curve](wandb_loss_curve.png)<img width="1752" height="899" alt="image" src="https://github.com/user-attachments/assets/a706252a-06b1-4cef-b4ed-eaf96920e26d" />


W&B run: https://wandb.ai/m-makpyr-iitu-edu-kz/marketing-finetune/runs/kg93esgc

Адаптер: сохранён только адаптер (не вся модель): 21.1 MB. Файлы: adapter_config.json, adapter_model.safetensors

---

Шаг 3 Оценка

Метрика: BERTScore. Выбрала BERTScore вместо ROUGE потому что маркетинговые тексты творческие точное совпадение слов не показывает реального качества. BERTScore учитывает семантическую близость через эмбеддинги.

Сравнивала ответы базовой и дообученной модели на 15 одинаковых промптах.

| Метрика | Значение |
|---|---|
| BERTScore Precision | 0.8706 |
| BERTScore Recall | 0.8750 |
| BERTScore F1 | 0.8727 |

```
[ 1] 0.8475 | Write a catchy Instagram caption for a new coffee brand...
[ 2] 0.9046 | Create a slogan for an eco-friendly water bottle.
[ 3] 0.8473 | Write a subject line for a promotional email about a 50% sale.
[ 4] 0.8874 | Write a product description for wireless noise-canceling headphones.
[ 5] 0.8619 | Create a Facebook ad for a fitness app targeting busy parents.
[ 6] 0.8743 | Write a tagline for a luxury skincare brand.
[ 7] 0.9027 | Write a promotional email for a new restaurant opening.
[ 8] 0.8745 | Create a Twitter post announcing a new product launch.
[ 9] 0.8802 | Write a call-to-action for a landing page selling online courses.
[10] 0.9053 | Create a marketing message for a back-to-school sale.
[11] 0.8688 | Write a headline for a travel agency summer campaign.
[12] 0.8598 | Create a slogan for a pet food brand focused on natural ingredients.
[13] 0.8473 | Write an email subject line for a flash sale ending tonight.
[14] 0.8948 | Create a product description for a smart home device.
[15] 0.8338 | Write a social media post for a charity fundraising campaign.
```

Ручной разбор: несмотря на близкие метрики (BERTScore ~0.84-0.85), модели ведут себя абсолютно по-разному.

Где дообученная лучше формат соцсетей: промпт на пост для благотворительности — базовая выдала сухое официальное письмо ("Dear friends...") и прилепила один хэштег для галочки. Дообученная отработала правильно: нормальная структура, релевантные хэштеги (#GivingTuesday), плейсхолдеры для картинок и чёткий CTA.

Где базовая справилась лучше короткие форматы: промпт на слоган для корма базовая честно выдала одну строчку. Дообученная проигнорировала ограничение формата вывалила огромный рекламный пост с emoji, который ещё и оборвался на полуслове.

Что пошло не так:
1. Format collapse - модель практически разучилась писать коротко
2. Эмоциональный передоз - везде клише ("Introducing...", "Hurry up!"), спам emoji (🎉, 🙌, 🔥)
3. Обрывы генерации - текст обрывается на полуслове, скорее всего неправильно настроен EOS-токен

Что буду фиксить в следующей итерации:
- Сбалансировать датасет - добавить больше коротких примеров
- Починить EOS-токены - проверить логику токенизации

---

Стек:
- Модель: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Метод: LoRA (r=16) через Hugging Face PEFT + Transformers
- Обучение: TRL SFTTrainer
- Логирование: Weights & Biases
- Оценка: BERTScore
- Платформа: Google Colab T4
