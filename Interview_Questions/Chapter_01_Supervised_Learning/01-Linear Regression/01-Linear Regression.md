

**یادگیری ماشین چیست و چه تفاوتی با برنامه‌نویسی سنتی دارد؟**

**پاسخ**: یادگیری ماشین شاخه‌ای از هوش مصنوعی است که به سیستم‌ها امکان می‌دهد از داده‌ها یاد بگیرند و عملکردشان را بدون برنامه‌نویسی صریح بهبود دهند. در برنامه‌نویسی سنتی، قوانین به‌صورت دستی کد می‌شوند، اما در یادگیری ماشین، مدل از داده‌ها الگوها را استخراج می‌کند.

---

**تعریف تام میچل از یادگیری ماشین چیست؟**

**پاسخ**: یک برنامه از تجربه $E$ نسبت به وظیفه $T$ و معیار عملکرد $P$ یاد می‌گیرد، اگر عملکردش در $T$، که با $P$ اندازه‌گیری می‌شود، با $E$ بهبود یابد.

---

**سوال: در تعریف تام ام. میچل از یادگیری ماشین، سه‌گانه $(T, P, E)$ را توضیح دهید. چگونه این سه جزء با یکدیگر ارتباط برقرار می‌کنند تا نشان دهند که یک برنامه یاد می‌گیرد؟**

**پاسخ**: سه‌گانه $(T, P, E)$ اجزای اصلی یک مسئله یادگیری را تشکیل می‌دهند:
**وظیفه (Task - T)**: کاری که برنامه کامپیوتری باید انجام دهد.
**معیار عملکرد (Performance Measure - P)**: معیاری برای اندازه‌گیری میزان خوب بودن عملکرد برنامه در انجام وظیفه $T$.
**تجربه (Experience - E)**: داده‌ها یا اطلاعاتی که برنامه برای یادگیری از آن‌ها استفاده می‌کند.
ارتباط: یک برنامه یاد می‌گیرد اگر عملکرد آن در وظایف $T$ که با $P$ اندازه‌گیری می‌شود، با افزایش تجربه $E$ بهبود یابد. این بدان معناست که با مشاهده داده‌های بیشتر یا با تعاملات مکرر (تجربه)، برنامه باید بتواند وظیفه محوله را با کیفیت بالاتری (عملکرد بهتر) انجام دهد.

---

**چند کاربرد عملی یادگیری ماشین را نام ببرید.**

**پاسخ**: پیش‌بینی رفتار مشتریان، کنترل کیفیت کارخانه، تحلیل تصاویر پزشکی.
---

**چه نوع‌هایی از یادگیری ماشین وجود دارد؟**

**پاسخ**: یادگیری ماشین به سه نوع اصلی تقسیم می‌شود:
**یادگیری نظارت‌شده**: مدل با استفاده از داده‌های برچسب‌دار (ورودی و خروجی) آموزش می‌بیند. مثلاً پیش‌بینی قیمت خانه بر اساس ویژگی‌هایی مانند متراژ و تعداد اتاق.
**یادگیری بدون نظارت**: مدل با استفاده از داده‌های بدون برچسب برای کشف الگوها یا ساختارهای پنهان در داده‌ها آموزش می‌بیند. مثلاً خوشه‌بندی مشتریان برای بازاریابی هدفمند.
**یادگیری تقویتی**: مدل از طریق آزمون و خطا و با دریافت پاداش یا جریمه یاد می‌گیرد. مثلاً آموزش یک ربات برای انجام وظایف خاص مانند حرکت در محیط.

---

**تفاوت بین یادگیری نظارت‌شده و بدون نظارت چیست؟**

**پاسخ**: در یادگیری نظارت‌شده، داده‌ها برچسب دارند (ورودی و خروجی مشخص)، اما در یادگیری بدون نظارت، داده‌ها بدون برچسب هستند و مدل الگوهای پنهان را کشف می‌کند.

---

**تفاوت بین supervised و unsupervised learning؟**

**پاسخ**: در supervised learning مدل با برچسب‌های آموزشی آموزش داده می‌شود، اما در unsupervised learning داده‌ها بدون برچسب هستند و مدل باید ساختار یا الگوهای پنهان در داده‌ها را کشف کند.

---

**یادگیری تحت نظارت چیست؟**

**پاسخ**: یادگیری با داده‌های برچسب‌دار که شامل ورودی $x$ و خروجی $y$ است برای پیش‌بینی خروجی‌های جدید.

---

**تفاوت بین رگرسیون و طبقه‌بندی چیست؟**

**پاسخ**: رگرسیون برای پیش‌بینی مقادیر پیوسته (مثل قیمت) و طبقه‌بندی برای پیش‌بینی دسته‌های گسسته (مثل اسپم/غیراسپم) است.

---

**رگرسیون خطی: مبانی و فرمول‌ها**

---

**رگرسیون خطی چیست و چگونه کار می‌کند؟**

**پاسخ**: رگرسیون خطی مدلی است که رابطه خطی بین متغیرهای مستقل و وابسته را مدل‌سازی می‌کند. معادله آن به‌صورت زیر است:
$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon$
هدف کمینه کردن خطا با یافتن ضرایب بهینه است.

---

**رگرسیون خطی؟**

**پاسخ**: رگرسیون خطی برای پیش‌بینی داده‌های پیوسته مانند قیمت‌ها، روندها و دما استفاده می‌شود.

---

**مدل ساده‌ای برای پیش‌بینی پیوسته چیست؟**

**پاسخ**: رگرسیون خطی یک مدل ساده برای پیش‌بینی داده‌های پیوسته است که رابطه‌ای خطی بین ورودی‌ها و خروجی برقرار می‌کند.

---

**فرمول فرضیه رگرسیون خطی چیست؟**

**پاسخ**:
$h_w(x) = w_0 + w_1 x_1 + \dots + w_D x_D = w^T x$

---

**معادله رگرسیون خطی را بنویسید و اجزای آن را توضیح دهید.**

**پاسخ**: معادله:
$y = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n + \epsilon$
$y$: متغیر وابسته (هدف).
$\beta_0$: عرض از مبدا.
$\beta_i$: ضرایب ویژگی‌ها.
$x_i$: متغیرهای مستقل (ویژگی‌ها).
$\epsilon$: خطای مدل.

---

**نقش $w_0$ در رگرسیون خطی چیست؟**

**پاسخ**: $w_0$ بایاس است که امکان پیش‌بینی مقادیر غیرصفر را حتی در صورت صفر بودن ویژگی‌ها فراهم می‌کند.

---

**نقش بایاس در رگرسیون چیست؟**

**پاسخ**: امکان جابجایی مدل برای تطابق بهتر با داده‌ها.

---

**هدف اصلی رگرسیون خطی چیست؟**

**پاسخ**: کمینه کردن فاصله بین پیش‌بینی $h_w(x)$ و مقدار واقعی $y$.

---

**چرا فرض خطی بودن در رگرسیون خطی مهم است؟**

**پاسخ**: رگرسیون خطی فرض می‌کند که رابطه بین متغیرهای ورودی و خروجی خطی است. اگر این فرض نقض شود، مدل نمی‌تواند روابط پیچیده‌تر داده‌ها را به درستی شبیه‌سازی کند و عملکرد ضعیفی خواهد داشت.

---

**اگر رابطه بین متغیرها غیرخطی باشد، چه باید کرد؟**

**پاسخ**: در صورت غیرخطی بودن رابطه بین متغیرها، می‌توان از تبدیل ویژگی‌ها (مانند افزودن توان‌ها یا لگاریتم‌ها) یا مدل‌های غیرخطی (مانند رگرسیون چندجمله‌ای یا درخت تصمیم) استفاده کرد.

---

**توابع هزینه و بهینه‌سازی**

---

**تابع هزینه چیست؟**

**پاسخ**: معیاری برای سنجش دقت مدل، مثل مجموع مربعات خطاها (SSE).

---

**تابع هزینه در رگرسیون خطی چیست و چگونه محاسبه می‌شود؟**

**پاسخ**: تابع هزینه معمولاً میانگین مربعات خطا (MSE) است:
$J(\beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
که $y_i$ مقدار واقعی و $\hat{y}_i$ مقدار پیش‌بینی‌شده است.

---

**فرمول MSE چیست؟**

**پاسخ**:
$J(w) = \frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - h_w(x^{(i)}))^2$

---

**چرا از MSE به‌عنوان تابع هزینه استفاده می‌شود؟**

**پاسخ**: چون خطاهای بزرگ را بیشتر جریمه می‌کند و محاسباتش ساده است.

---

**چرا MSE خطاهای بزرگ را بیشتر جریمه می‌کند؟**

**پاسخ**: چون خطاها را به توان 2 می‌رساند، تأثیر خطاهای بزرگ بیشتر می‌شود.

---

**تعریف MSE؟**

**پاسخ**: MSE یا میانگین مربعات خطا، میانگین مربعات اختلاف بین پیش‌بینی‌ها و مقدار واقعی است.

---

**تفاوت MAE و MSE؟**

**پاسخ**: MAE (میانگین مطلق خطا) خطاها را به‌صورت خطی اندازه‌گیری می‌کند، در حالی که MSE (میانگین مربعات خطا) خطاهای بزرگ‌تر را بیشتر جریمه می‌کند.

---

**چرا تابع MSE مشتق‌پذیر است و MAE نه؟**

**پاسخ**: MSE (میانگین مربعات خطا) تابعی مربعی است که در تمام نقاط مشتق‌پذیر است، اما MAE (میانگین مطلق خطا) در نقطه صفر مشتق ندارد، زیرا به صورت خطی و بدون انحنا است.

---

**چه زمانی استفاده از MAE به جای MSE منطقی‌تر است؟**

**پاسخ**: MAE (میانگین قدرمطلق خطاها) زمانی بهتر است که داده‌ها شامل outlier باشند، زیرا MAE خطاها را به صورت خطی جریمه می‌کند و برخلاف MSE که به شدت از outlier تأثیر می‌گیرد، کمتر به خطاهای بزرگ حساس است.
در نتیجه، MAE برای داده‌هایی که دارای داده‌های پرت هستند، بهتر عمل می‌کند و ممکن است نمایانگر عملکرد واقعی‌تر مدل باشد.

---

**چرا MSE نسبت به خطاهای بزرگ حساس‌تر است؟**

**پاسخ**: چون در MSE از مربع خطاها استفاده می‌شود، خطاهای بزرگ بیش از حد تقویت می‌شوند. به عبارت دیگر، هرچه خطا بزرگ‌تر باشد، تأثیر بیشتری بر MSE دارد.
فرمول MSE به صورت زیر است:
$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
این ویژگی باعث می‌شود که اگر داده‌های شما شامل outlier (داده‌های پرت) باشند، MSE ممکن است نمایانگر عملکرد واقعی مدل نباشد. در این شرایط، استفاده از MAE یا Huber Loss ممکن است مناسب‌تر باشد.

---

**روش تحلیلی در رگرسیون خطی چیست؟**

**پاسخ**: استفاده از معادلات نرمال برای محاسبه مستقیم
$w = (X^T X)^{-1} X^T y$

---

**مزیت روش تحلیلی چیست؟**

**پاسخ**: دقیق است و نیازی به تکرار ندارد.

---

**محدودیت‌های روش معادلات نرمال چیست؟**

**پاسخ**: محاسبات سنگین برای داده‌های بزرگ و نیاز به معکوس‌پذیری $X^T X$.

---

**چرا $X^T X$ ممکن است معکوس‌پذیر نباشد؟**

**پاسخ**: به دلیل هم‌خطی ویژگی‌ها یا تعداد کم نمونه‌ها.

---

**گرادیان دیسنت (Gradient Descent)**

---

**گرادیان نزولی چیست؟**

**پاسخ**: روشی عددی برای کمینه کردن تابع هزینه با به‌روزرسانی وزن‌ها در جهت مخالف گرادیان.

---

**گرادیان دیسنت چیست؟**

**پاسخ**: گرادیان دیسنت الگوریتمی برای کمینه‌سازی تابع هزینه است که در هر مرحله با استفاده از گرادیان، پارامترهای مدل به‌روز می‌شود.

---

**گرادیان چیست؟**

**پاسخ**: مشتق تابع هزینه نسبت به پارامترها که جهت بهینه‌سازی را نشان می‌دهد.

---

**فرمول به‌روزرسانی گرادیان نزولی چیست؟**

**پاسخ**:
$w_{t+1} = w_t - \eta \nabla J(w_t)$

---

**نرخ یادگیری ($\eta$) چیست؟**

**پاسخ**: پارامتری که اندازه قدم‌های به‌روزرسانی وزن‌ها را تعیین می‌کند.

---

**اگر نرخ یادگیری خیلی بزرگ باشد چه اتفاقی می‌افتد؟**

**پاسخ**: الگوریتم ممکن است واگرا شود و به نقطه بهینه نرسد.

---

**اگر نرخ یادگیری خیلی کوچک باشد چه می‌شود؟**

**پاسخ**: همگرایی کند می‌شود و زمان زیادی طول می‌کشد.

---

**چه زمانی گرادیان دیسنت همگرا نمی‌شود؟**

**پاسخ**: گرادیان دیسنت ممکن است زمانی که نرخ یادگیری زیاد باشد یا تابع هزینه غیرمحدب باشد، همگرا نشود.

---

**چگونه نرخ یادگیری مناسب انتخاب می‌شود؟**

**پاسخ**: با آزمایش مقادیر مختلف یا استفاده از روش‌های تطبیقی مثل Adam.

---

**تفاوت Batch GD و Stochastic GD چیست؟**

**پاسخ**: Batch GD از کل داده‌ها و Stochastic GD از یک نمونه در هر مرحله استفاده می‌کند.

---

**چه زمانی از Batch GD استفاده می‌کنیم؟**

**پاسخ**: وقتی داده‌ها کم باشند و دقت بالا مهم باشد.

---

**چه زمانی Stochastic GD مناسب است؟**
**پاسخ**: برای داده‌های بزرگ یا مسائل آنلاین که سرعت مهم است.

---

**Mini-batch GD چیست؟**

**پاسخ**: استفاده از زیرمجموعه‌ای از داده‌ها برای به‌روزرسانی وزن‌ها، تعادل بین دقت و سرعت.

---

**مزیت Mini-batch GD چیست؟**

**پاسخ**: تعادل بین سرعت Stochastic GD و دقت Batch GD.

---

**چرا نرمال‌سازی داده‌ها در گرادیان نزولی مهم است؟**

**پاسخ**: باعث می‌شود گرادیان‌ها در مقیاس مشابه باشند و همگرایی سریع‌تر شود.

---

**چرا نرمال‌سازی داده‌ها در رگرسیون خطی مهم است؟**

**پاسخ**: نرمال‌سازی مقیاس ویژگی‌ها را یکسان می‌کند تا تأثیر متغیرهای با مقیاس بزرگ‌تر بر مدل کاهش یابد و گرادیان کاهشی سریع‌تر همگرا شود.

---

**گرادیان کاهشی چیست و چگونه در رگرسیون خطی استفاده می‌شود؟**

**پاسخ**: گرادیان کاهشی الگوریتمی برای کمینه کردن تابع هزینه است. در رگرسیون خطی، ضرایب مدل با به‌روزرسانی‌های تکراری در جهت کاهش گرادیان تابع هزینه تنظیم می‌شوند.

---

### **رگرسیون چندجمله‌ای و مشکلات مدل**

---

**رگرسیون چندجمله‌ای چیست؟**

**پاسخ**: مدلی که روابط غیرخطی را با استفاده از ویژگی‌های چندجمله‌ای مدل می‌کند.

---

**فرمول فرضیه رگرسیون چندجمله‌ای چیست؟**

**پاسخ**:
$h(x) = w_0 + w_1 x + w_2 x^2 + \dots + w_m x^m$

---

**مزیت رگرسیون چندجمله‌ای نسبت به رگرسیون خطی چیست؟**

**پاسخ**: توانایی مدل‌سازی روابط غیرخطی.

---

**چرا از Polynomial Regression استفاده می‌شود؟**

**پاسخ**: Polynomial Regression برای مدل‌سازی روابط غیرخطی میان متغیرهای ورودی و خروجی به کار می‌رود. این روش به وسیله افزودن توان‌های مختلف به ویژگی‌ها، قادر است الگوهای پیچیده‌تری از داده‌ها را شبیه‌سازی کند.

---

**چرا رگرسیون چندجمله‌ای ممکن است بیش‌برازش کند؟**

**پاسخ**: چون با افزایش درجه، مدل ممکن است نویز داده‌ها را هم یاد بگیرد.

---

**Underfitting چیست؟**

**پاسخ**: وقتی مدل بیش‌ازحد ساده است و نمی‌تواند الگوهای داده را خوب یاد بگیرد.

---

**Overfitting چیست؟**

**پاسخ**: وقتی مدل بیش‌ازحد پیچیده است و نویز داده‌ها را هم یاد می‌گیرد.

---

**چگونه Overfitting را تشخیص دهیم؟**

**پاسخ**: Overfitting زمانی رخ می‌دهد که خطای آموزش پایین و خطای تست بالا باشد، زیرا مدل فقط به داده‌های آموزش مناسب شده است و توانایی تعمیم به داده‌های جدید را ندارد.

---

**چه زمانی underfitting داریم؟**

**پاسخ**: Underfitting زمانی اتفاق می‌افتد که مدل ساده باشد و نتواند الگوهای پیچیده داده‌ها را به درستی یاد بگیرد.

---

**تفاوت بین بیش‌برازش و کم‌برازش چیست؟**

**پاسخ**:
**بیش‌برازش**: مدل بیش از حد به داده‌های آموزشی وابسته است و روی داده‌های جدید ضعیف عمل می‌کند.
**کم‌برازش**: مدل الگوهای داده‌های آموزشی را به‌خوبی یاد نمی‌گیرد و عملکرد ضعیفی دارد.

---

**چرا مدل‌های پیچیده overfit می‌کنند؟**

**پاسخ**: مدل‌های پیچیده قادرند حتی نویز داده را یاد بگیرند، که باعث می‌شود نتوانند داده‌های جدید را به خوبی تعمیم دهند و دچار overfitting شوند.

---

**چرا رگرسیون خطی برای داده‌های غیرخطی مناسب نیست؟**

**پاسخ**: چون فرض می‌کند رابطه بین ورودی و خروجی خطی است.

---

### **تعمیم‌پذیری و ارزیابی مدل**

---

**تعمیم‌پذیری در یادگیری ماشین به چه معناست؟**

**پاسخ**: تعمیم‌پذیری توانایی مدل در عملکرد خوب روی داده‌های جدید و نادیده است، نه فقط داده‌های آموزشی.

---

**چگونه می‌توان تعمیم‌پذیری یک مدل را ارزیابی کرد؟**

**پاسخ**: با استفاده از مجموعه آزمون جداگانه، معیارهایی مانند MSE یا دقت، و روش‌هایی مانند اعتبارسنجی متقاطع.

---

**چگونه می‌توان مدل رگرسیون را ارزیابی کرد؟**

**پاسخ**: با معیارهایی مثل MSE، RMSE، یا $R^2$ روی داده‌های تست.

---

**چگونه می‌توان مدل رگرسیون خطی را ارزیابی کرد؟**

**پاسخ**: برای ارزیابی مدل رگرسیون خطی از معیارهایی مانند MSE (میانگین مربعات خطا)، RMSE (ریشه میانگین مربعات خطا)، و $R^2$ استفاده می‌شود. همچنین، بررسی نمودارهای باقی‌مانده می‌تواند به ارزیابی تطابق مدل با داده‌های واقعی کمک کند.

---

**تفاوت MSE و RMSE چیست؟**

**پاسخ**: RMSE ریشه دوم MSE است و مقیاسش با داده‌ها یکسان است.

---

**تفاوت بین MSE و RMSE چیست؟**

**پاسخ**:
RMSE (ریشه میانگین مربعات خطا) از MSE (میانگین مربعات خطا) به‌صورت ریشه دوم محاسبه می‌شود:
$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
و مقیاس آن با داده‌های اصلی هم‌خوانی دارد، که تفسیر آن را برای استفاده‌های عملی ساده‌تر می‌کند.

---

**چرا از Root Mean Squared Error استفاده می‌کنیم؟**

**پاسخ**: RMSE (ریشه میانگین مربعات خطا) به دلیل اینکه با واحد‌های خروجی مدل یکسان است، تفسیر راحت‌تری دارد. همچنین این معیار خطاهای بزرگ را بیشتر جریمه می‌کند که به مدل کمک می‌کند تا دقت بیشتری در پیش‌بینی‌ها داشته باشد.


---

**در تفسیر Root Mean Squared Error، چرا گفتیم واحد آن با خروجی برابر است؟**

**پاسخ کامل**:
چون RMSE برابر است با:
$\text{RMSE} = \sqrt{\text{MSE}}$
از آنجایی که MSE واحدی به شکل (خروجی)² دارد، با گرفتن ریشه دوم از آن، واحد RMSE با واحد خروجی یکسان می‌شود. این ویژگی باعث می‌شود که RMSE تفسیر‌پذیرتر باشد و بگوییم "مدل به طور میانگین ۲ درجه خطا دارد."

---

**آیا کاهش MSE همیشه خوب است؟**

**پاسخ**:
نه، کاهش MSE بر روی داده‌های آموزش می‌تواند باعث overfitting شود، اگر مدل نتواند به خوبی داده‌های تست را پیش‌بینی کند.

---

**تفاوت داده‌های آموزشی و اعتبارسنجی چیست؟**

**پاسخ**:
داده‌های آموزشی برای یادگیری مدل و اعتبارسنجی برای ارزیابی آن استفاده می‌شوند.

---

**داده‌های اعتبارسنجی چه نقشی دارند؟**

**پاسخ**:
برای انتخاب مدلی که نه خیلی ساده و نه خیلی پیچیده باشد.

---

**چه تفاوتی بین Validation و Test Set وجود دارد؟**

**پاسخ**:
Validation Set برای تنظیم مدل، انتخاب بهترین هایپرپارامترها و جلوگیری از بیش‌برازش استفاده می‌شود، در حالی که Test Set برای ارزیابی نهایی و بررسی کارایی واقعی مدل استفاده می‌شود.

---

**چگونه درجه مناسب در رگرسیون چندجمله‌ای انتخاب میشود؟**

**پاسخ**:
با استفاده از داده‌های اعتبارسنجی و بررسی خطای تست.

---

**چرا برای ارزیابی عملکرد مدل‌های یادگیری ماشین، استفاده از مجموعه تست (Test Set) جدا از مجموعه آموزشی (Training Set) ضروری است؟**

**پاسخ**:
استفاده از مجموعه تست جداگانه از مجموعه آموزشی برای ارزیابی مدل ضروری است زیرا هدف اصلی یادگیری ماشین، ارزیابی توانایی مدل در تعمیم دادن به داده‌های جدید است. اگر داده‌های تست در فرآیند آموزش به کار روند، مدل ممکن است فقط داده‌های خاص مجموعه تست را حفظ کند و به خوبی بر روی داده‌های جدید عمل نکند. بنابراین، استفاده از مجموعه تست برای سنجش عملکرد مدل بر روی داده‌های غیرمشاهده‌شده و ناشناخته بسیار مهم است.

---

### **Bias-Variance Tradeoff**

---

**تفاوت بین واریانس و بایاس در یادگیری ماشین چیست؟**

**پاسخ**:
**بایاس**: خطای ناشی از ساده‌سازی بیش از حد مدل، که باعث می‌شود مدل نتواند الگوهای پیچیده‌تر داده‌ها را به‌درستی یاد بگیرد.
**واریانس**: حساسیت مدل به تغییرات جزئی در داده‌های آموزشی، که ممکن است مدل را در برابر داده‌های جدید حساس و ضعیف کند.

---

**تفاوت Bias و Variance چیست؟**

**پاسخ**:
**Bias** خطای ناشی از ساده بودن مدل است که مدل نمی‌تواند الگوهای پیچیده داده‌ها را به درستی شبیه‌سازی کند.
**Variance** حساسیت مدل به تغییرات داده‌های آموزشی است که ممکن است باعث شود مدل به داده‌های جدید تعمیم نپذیرد.

---

**چگونه می‌توان تعادل بین واریانس و بایاس را برقرار کرد؟**

**پاسخ**:
برای حفظ تعادل بین بایاس و واریانس، باید مدلی با پیچیدگی مناسب انتخاب کرد. استفاده از منظم‌سازی، اعتبارسنجی متقاطع، و دقت در انتخاب ویژگی‌ها می‌تواند به این تعادل کمک کند.

---

**Bias-Variance Tradeoff چیست؟**
**پاسخ**:

در یادگیری ماشین، خطای تعمیم‌پذیری به سه جزء تقسیم می‌شود:

* **Bias**: خطای ناشی از ساده بودن مدل
* **Variance**: نوسان مدل نسبت به تغییر داده آموزش
* **Noise**: خطای ذاتی داده

**Tradeoff** یعنی افزایش پیچیدگی باعث کاهش Bias ولی افزایش Variance می‌شود. هنر طراحی مدل، رسیدن به تعادل بهینه بین این دو است.

---

**چرا Bias² و Variance جمع می‌شوند؟**

**پاسخ**:
Bias² و Variance دو منبع اصلی خطای کلی مدل هستند. Bias به خطای ناشی از فرضیات غلط مدل و Variance به حساسیت مدل به تغییرات داده‌های آموزشی اشاره دارد. خطای کلی مدل برابر با مجموع این دو است.

---

**چگونه Bias بالا را کاهش دهیم؟**

**پاسخ**:
برای کاهش Bias، می‌توان از مدل‌های پیچیده‌تر یا افزایش ویژگی‌ها استفاده کرد تا مدل قادر باشد الگوهای پیچیده‌تر داده‌ها را یاد بگیرد.

---

**اگر مدل‌تان Bias بالا دارد، چه اقداماتی انجام می‌دهید؟**

**پاسخ**:
مدلی که Bias بالا دارد، معمولاً به دلیل سادگی زیادش قادر به یادگیری روابط پیچیده داده‌ها نیست. برای کاهش Bias می‌توان از روش‌های زیر استفاده کرد:

* افزایش پیچیدگی مدل (مثلاً استفاده از درجه بالاتر در رگرسیون چندجمله‌ای)
* اضافه کردن ویژگی‌های جدید به داده‌ها
* استفاده از مدل‌های غیرخطی مانند درخت تصمیم یا شبکه عصبی که قادر به مدل‌سازی روابط پیچیده‌تری هستند.

---

**چگونه Variance بالا را کاهش دهیم؟**

**پاسخ**:
برای کاهش Variance، می‌توان از Regularization، استفاده از داده‌های بیشتر، یا مدل‌های ساده‌تر استفاده کرد تا مدل کمتر به داده‌های آموزشی حساس باشد.

---

**اگر مدل‌تان Variance بالا دارد، چگونه با آن مقابله می‌کنید؟**

**پاسخ**:
مدل‌هایی با Variance بالا به تغییرات داده‌های آموزشی حساس هستند و معمولاً به Overfitting منجر می‌شوند. برای مقابله با این مشکل، می‌توان از روش‌های زیر استفاده کرد:

* استفاده از Regularization مانند Ridge یا Lasso که مدل را از پیچیده شدن بیش از حد باز می‌دارد.
* افزایش حجم داده آموزشی برای کاهش حساسیت مدل به داده‌های خاص.
* ساده‌تر کردن مدل (مثلاً کاهش درجه رگرسیون چندجمله‌ای).
* استفاده از Cross-validation برای ارزیابی مدل روی داده‌های مختلف و جلوگیری از overfitting.

---

**تعریف دقیق Noise در Bias-Variance چیست؟**


**پاسخ**:
Noise بخشی از خطاست که به‌دلیل تصادفی بودن داده‌ها ایجاد می‌شود و قابل کاهش نیست. این خطا از طبیعت داده‌ها ناشی می‌شود و نمی‌توان آن را حذف کرد.

---

**چرا نمی‌توان هردو (Bias و Variance) را هم‌زمان به حداقل رساند؟**


**پاسخ**:
Bias و Variance دو مؤلفه متضاد هستند:

* مدل‌های ساده (مثلاً رگرسیون خطی) معمولاً Bias بالا دارند، زیرا قادر به یادگیری الگوهای پیچیده نیستند.
* مدل‌های پیچیده (مثلاً شبکه‌های عصبی عمیق یا رگرسیون چندجمله‌ای با درجه بالا) Variance بالا دارند، زیرا نسبت به تغییرات داده‌های آموزشی حساس هستند.

کاهش یکی از این دو معمولاً منجر به افزایش دیگری می‌شود. بنابراین، باید بین این دو یک تعادل هوشمندانه برقرار کرد که به آن **Bias-Variance Tradeoff** گفته می‌شود.

---

سوالات کلی و تفکربرانگیز

---

### **رگرسیون خطی در چه مواردی استفاده می‌شود؟**

**پاسخ**:
رگرسیون خطی برای پیش‌بینی مقادیر عددی در مسائلی مانند:

* پیش‌بینی فروش محصولات بر اساس بودجه تبلیغات
* تخمین مصرف انرژی یک ساختمان بر اساس دما و اندازه
* تحلیل روابط اقتصادی، مانند تأثیر نرخ بهره بر رشد اقتصادی

برای بهبود دقت مدل، می‌توان از تکنیک‌هایی مانند نرمال‌سازی داده‌ها، انتخاب ویژگی‌های مناسب و استفاده از روش‌های منظم‌سازی (مانند رگرسیون ریج یا لاسو) استفاده کرد.

---

### **چه مشکلاتی در استفاده از رگرسیون خطی ممکن است پیش بیاید؟**

**پاسخ**:
چند مشکل اصلی عبارت‌اند از:

* **بیش‌برازش (Overfitting)**: مدل ممکن است به‌طور دقیق داده‌های آموزشی را یاد بگیرد اما در داده‌های جدید عملکرد ضعیفی داشته باشد.
* **فرض خطی بودن**: اگر رابطه بین متغیرها غیرخطی باشد، رگرسیون خطی قادر به مدل‌سازی دقیق نخواهد بود.
* **حساسیت به داده‌های پرت**: داده‌های پرت می‌توانند ضرایب مدل را به‌شدت تحت تأثیر قرار دهند.

---

### **چرا تعمیم‌پذیری در یادگیری ماشین مهم است؟**

**پاسخ**:
تعمیم‌پذیری به این معناست که مدل فقط به داده‌های آموزشی وابسته نباشد، بلکه قادر باشد روی داده‌های جدید (داده‌های آزمون) نیز به‌خوبی عمل کند. هدف اصلی یادگیری ماشین، ساخت مدل‌هایی است که الگوهای عمومی را از داده‌ها استخراج کنند و نه اینکه فقط داده‌های آموزشی را حفظ کنند. اگر مدل نتواند به‌خوبی تعمیم دهد، ممکن است دچار بیش‌براش یا کم‌براش شود.

---

### **چگونه می‌توان مدل‌های پیچیده را برای جلوگیری از بیش‌براش تنظیم کرد؟**

**پاسخ**:
برای جلوگیری از بیش‌براش در مدل‌های پیچیده، می‌توان از استراتژی‌های زیر استفاده کرد:

* **رگولاریزاسیون**: اضافه کردن جریمه به وزن‌ها (با استفاده از L1 یا L2 رگولاریزاسیون) باعث می‌شود که مدل نتواند به جزئیات نویزی داده‌های آموزشی وابسته شود.
* **کاهش پیچیدگی مدل**: کاهش تعداد ویژگی‌ها یا پارامترهای مدل می‌تواند از پیچیدگی بیش از حد جلوگیری کند. به عنوان مثال، استفاده از پرسپترون‌های ساده‌تر یا شبکه‌های عصبی با تعداد لایه‌های کمتر می‌تواند مدل را ساده‌تر کند.
* **افزایش داده‌ها**: مدل‌ها معمولاً هنگامی که داده‌های بیشتری برای آموزش دارند، بهتر عمل می‌کنند. افزایش داده‌ها به کاهش خطر بیش‌براش کمک می‌کند.
* **انتخاب ویژگی‌ها**: استفاده از تکنیک‌های انتخاب ویژگی‌های مرتبط و حذف ویژگی‌های غیرمفید یا نویزی می‌تواند پیچیدگی مدل را کاهش دهد و از مدل‌سازی جزئیات غیرضروری جلوگیری کند.

---

### **چرا باید از مجموع مربعات خطا (SSE) به عنوان تابع هزینه استفاده کنیم و چرا این معیار در رگرسیون خطی پرکاربرد است؟**

**پاسخ**:
مجموع مربعات خطا (SSE) به دلیل ویژگی‌های خاص خود یکی از معیارهای پرکاربرد در رگرسیون خطی است:

* **سادگی و شفافیت**: SSE، تفاوت بین مقادیر واقعی و پیش‌بینی شده را محاسبه کرده و آن را مربع می‌کند تا از تأثیر تفاوت‌های بزرگتر جلوگیری شود. این کار باعث می‌شود پیش‌بینی‌های نادرست بزرگ‌تر جریمه بیشتری داشته باشند.
* **قابلیت بهینه‌سازی آسان**: SSE به راحتی قابل بهینه‌سازی است. با استفاده از الگوریتم‌های بهینه‌سازی مبتنی بر گرادیان نزولی، می‌توان این تابع هزینه را کمینه کرد.
* **ارتباط با MLE**: کمینه‌سازی SSE معادل بیشینه‌سازی **log-likelihood** در رگرسیون احتمالی است. از این منظر آماری، استفاده از SSE در رگرسیون خطی کاملاً منطقی است.

---

### **آیا همیشه از رگرسیون خطی برای مدل‌سازی داده‌ها استفاده می‌کنیم؟ چه زمانی باید به روش‌های دیگری فکر کنیم؟**

**پاسخ**:
رگرسیون خطی معمولاً زمانی مفید است که رابطه بین ورودی‌ها و خروجی‌ها تقریباً خطی باشد. اما اگر رابطه‌ها غیرخطی باشند، رگرسیون خطی ممکن است به کم‌براش یا بیش‌براش منجر شود.
در این شرایط، می‌توان از روش‌های دیگر استفاده کرد:

* **رگرسیون غیرخطی**: استفاده از مدل‌های رگرسیونی که می‌توانند روابط غیرخطی را مدل‌سازی کنند.
* **درخت‌های تصمیم (Decision Trees)**: برای مدل‌سازی روابط پیچیده‌تر از درخت‌های تصمیم استفاده می‌شود که قادرند داده‌ها را به شکل غیرخطی تقسیم کنند.
* **شبکه‌های عصبی (Neural Networks)**: این مدل‌ها قادرند روابط پیچیده و غیرخطی بین ورودی‌ها و خروجی‌ها را یاد بگیرند و برای داده‌های پیچیده مناسب‌تر هستند.

---

### **Bias-Variance Tradeoff چیست و چگونه می‌توان آن را در انتخاب مدل‌های یادگیری ماشین به کار برد؟**

**پاسخ**:
**Bias-Variance Tradeoff** به چالشی اشاره دارد که باید بین **بایاس** (Bias) و **واریانس** (Variance) تعادل برقرار کرد:

* **بایاس بالا**: نشان‌دهنده مدل‌های ساده‌ای است که قادر به مدل‌سازی پیچیدگی‌های داده‌ها نیستند و معمولاً منجر به **کم‌براش** می‌شوند.
* **واریانس بالا**: نشان‌دهنده مدل‌هایی است که به شدت به داده‌های آموزشی وابسته هستند و نمی‌توانند به خوبی بر روی داده‌های جدید تعمیم یابند. این موضوع به **بیش‌براش** مدل منجر می‌شود.

برای دستیابی به بهترین عملکرد، باید پیچیدگی مدل را به نحوی تنظیم کرد که بایاس و واریانس به طور متعادل به حداقل برسند.

---

### **در رگرسیون خطی، چرا مدل ممکن است در برابر داده‌های جدید ناتوان باشد حتی اگر خطای آموزش بسیار کم باشد؟**

**پاسخ**:
اگر مدل دارای **بیش‌براش** باشد (یعنی خطای آموزش بسیار کم و خطای تست بالا باشد)، احتمالاً مدل دچار پیچیدگی اضافی شده است که باعث شده است به جای یادگیری الگوهای عمومی، تنها به جزئیات و نویز داده‌های آموزشی توجه کند. در این حالت، مدل به داده‌های جدید تعمیم خوبی نخواهد داشت، زیرا توانایی آن در تشخیص الگوهای واقعی کاهش یافته است. این اتفاق معمولاً در مدل‌های پیچیده‌تر مانند رگرسیون چندجمله‌ای با درجه بالا یا شبکه‌های عصبی با تعداد زیاد لایه‌ها رخ می‌دهد که می‌توانند بیش از حد به داده‌های آموزشی فیت شوند.

---

### **سوال: چه عواملی می‌تواند باعث شود که خطای آموزش پایین باشد ولی خطای تست بالا باشد؟**

**پاسخ**:
این وضعیت معمولاً نشان‌دهنده **بیش‌براش** است. عوامل مختلفی می‌توانند باعث این مشکل شوند:

* **مدل پیچیده** با تعداد زیاد پارامتر که قادر به یادگیری جزئیات و نویزهای داده‌های آموزشی است.
* **پیش‌پردازش نامناسب داده‌ها**، مانند نرمال‌سازی ناقص یا ویژگی‌های غیرضروری که به مدل اضافه می‌شوند.
* **استفاده از مدل‌هایی که بیش از حد به داده‌های آموزشی وابسته هستند** (مثل رگرسیون چندجمله‌ای با درجه بالا).

در چنین شرایطی، مدل در برابر داده‌های جدید عملکرد ضعیفی دارد چون نمی‌تواند الگوهای عمومی را یاد بگیرد و تنها به خصوصیات خاص داده‌های آموزشی توجه می‌کند.

---

### **سوال: MSE (میانگین مربعات خطا) و RMSE (ریشه میانگین مربعات خطا) چه تفاوت‌هایی دارند و در ارزیابی مدل‌ها کدام یک بهتر است؟**

**پاسخ**:

* **MSE** میانگین مربعات خطاها را محاسبه می‌کند. این معیار نشان‌دهنده متوسط تفاوت‌های مربعی بین پیش‌بینی‌ها و مقادیر واقعی است.
* \*\*RMSE


\*\* ریشه مربع **MSE** است و به واحدهای داده نزدیک‌تر است. RMSE برای مدل‌هایی که نیاز به درک تفاوت‌های واقعی در واحدهای داده دارند، مفیدتر است.

اگرچه هر دو معیار اطلاعات مشابهی ارائه می‌دهند، **RMSE** برای تحلیل‌های خاص که نیاز به تفکیک تفاوت‌های خطا در مقیاس واقعی دارند، مناسب‌تر است.

---


### **چه زمانی از Polynomial Regression استفاده می‌کنیم و خطر آن چیست؟**

**پاسخ**:
وقتی رابطه‌ی بین ورودی و خروجی **غیرخطی** است، از رگرسیون چندجمله‌ای استفاده می‌کنیم.
خطر آن در این است که **درجه بالا** → **Overfitting** چون مدل می‌تواند به راحتی روی نویز داده آموزش هم منطبق شود.

---

### **تعریف دقیق Expected Test Error چیست؟**

**پاسخ**:
**Expected Test Error** یعنی میانگین خطای مدل روی داده‌های جدیدی که از همان توزیع $p(x,y)$ استخراج شده‌اند:

$$
J(w)=E_{(x,y)∼p}[(y−h_w(x))^2]
$$

از آنجا که توزیع اصلی را نداریم، این مقدار با داده‌های تست تخمین زده می‌شود.

---

### **Root Mean Squared Error (RMSE) چه تفاوتی با MSE دارد و چه زمانی استفاده می‌شود؟**

**پاسخ**:

* **MSE** فقط میانگین مربعات خطا را گزارش می‌کند و واحد آن مربع واحد خروجی است.
* **RMSE** ریشه‌ی **MSE** است و واحد آن مثل خروجی اصلی است.

مثلاً اگر داریم قیمت خانه‌ها را پیش‌بینی می‌کنیم، RMSE در واحد «تومان» خواهد بود و قابل تفسیرتر است. بنابراین RMSE اغلب برای گزارش به کاربران یا ارزیابی عملکرد کاربردی ترجیح داده می‌شود.

---

### **چرا مدل‌های بسیار ساده (High Bias) حتی با داده زیاد هم عملکرد ضعیفی دارند؟**

**پاسخ**:
مدل‌های ساده دارای فرضیات محدودکننده‌ای هستند. مثلاً در رگرسیون خطی، ما فرض می‌کنیم رابطه بین ورودی و خروجی **خطی** است. اگر واقعیت پیچیده‌تر باشد، حتی اگر حجم داده زیاد باشد، این مدل قادر به یادگیری آن نخواهد بود.
این خطای سیستماتیک را **Bias** می‌گویند. داده بیشتر نمی‌تواند فرض غلط مدل را جبران کند.

---

### **آیا همیشه باید از مدل‌های پیچیده برای بهبود عملکرد استفاده کرد؟**

**پاسخ**:
**خیر!** مدل پیچیده لزوماً بهتر نیست. پیچیدگی بالا اگر با داده ناکافی یا تنظیمات نامناسب همراه شود، **Overfitting** ایجاد می‌کند.
برعکس، مدل ساده با انتخاب ویژگی مناسب، **Regularization** و تنظیم پارامتر صحیح می‌تواند عملکرد بهتری از مدل پیچیده داشته باشد.
**اصل طلایی**: مدل باید به اندازه‌ی لازم پیچیده باشد، نه بیشتر.

---

### **چگونه می‌توان به طور عملی و با استفاده از ارزیابی‌های مختلف، بهترین مدل را در رگرسیون انتخاب کرد؟**

**پاسخ**:
برای انتخاب بهترین مدل در رگرسیون، می‌توان از ارزیابی‌های مختلفی استفاده کرد:

* **Cross-Validation**: اعتبارسنجی متقابل به کاهش واریانس ارزیابی کمک می‌کند و مدل‌ها را روی داده‌های مختلف می‌سنجد.
* **مجموعه تست**: ارزیابی مدل بر روی داده‌های دیده‌نشده برای سنجش قدرت تعمیم مدل ضروری است.
* **معیارهای ارزیابی**: از جمله **MSE** (میانگین مربعات خطا)، **RMSE** (ریشه میانگین مربعات خطا)، **R²** (ضریب تعیین) که میزان انطباق مدل با داده‌ها را نشان می‌دهند.
* **آزمون‌های آماری**: مانند آزمون‌های **F** و **t** برای بررسی میزان اعتبار پارامترهای مدل.

با مقایسه این معیارها، مدل‌هایی که کمترین خطا و بهترین تعمیم‌پذیری را دارند، انتخاب می‌شوند.

---

### **چرا رگرسیون خطی در مواقعی که روابط پیچیده‌تری بین متغیرها وجود دارد، ممکن است نتیجه دقیقی ندهد؟**

**پاسخ**:
رگرسیون خطی فرض می‌کند که رابطه بین متغیرهای ورودی و خروجی **خطی** است. این فرض ممکن است در بسیاری از موارد عملی صدق نکند. هنگامی که روابط پیچیده‌تری میان ورودی‌ها و خروجی وجود دارد (مثلاً روابط **غیرخطی**)، رگرسیون خطی قادر به مدل‌سازی این روابط نیست و باعث **کم‌براش** می‌شود. در این مواقع، مدل‌های پیچیده‌تری مانند **رگرسیون چندجمله‌ای** یا **شبکه‌های عصبی** که قابلیت مدل‌سازی روابط غیرخطی را دارند، می‌توانند عملکرد بهتری داشته باشند.

---

### **در رگرسیون چندجمله‌ای، چرا استفاده از درجه‌های بسیار بالا (مانند $M \geq 10$ ) ممکن است منجر به از دست دادن قدرت تعمیم مدل شود؟**

**پاسخ**:
در رگرسیون چندجمله‌ای با درجات بالا، مدل به طور فزاینده‌ای پیچیده می‌شود و سعی می‌کند به تمام نوسانات داده‌های آموزشی، حتی نویزهای تصادفی، واکنش نشان دهد. این پدیده منجر به **بیش‌براش** می‌شود، یعنی مدل به جای یادگیری الگوهای عمومی، به جزئیات و نویزهای موجود در داده‌ها فیت می‌شود. نتیجه این است که مدل عملکرد ضعیفی در برابر داده‌های جدید خواهد داشت، چرا که توانایی تعمیم به داده‌های دیده‌نشده کاهش می‌یابد.

---

### **در رگرسیون خطی، چگونه می‌توان متوجه شد که مدل بیش از حد پیچیده یا ساده است؟**

**پاسخ**:
برای ارزیابی پیچیدگی مدل در رگرسیون خطی، می‌توان به خطای آموزش و خطای تست نگاه کرد:

* اگر **خطای آموزش** و **خطای تست** هر دو بالا باشند، مدل **کم‌براش** است. این بدان معناست که مدل نتوانسته است رابطه واقعی موجود در داده‌ها را یاد بگیرد و نیاز به پیچیده‌تر شدن دارد.
* اگر **خطای آموزش** پایین و **خطای تست** بالا باشد، مدل **بیش‌براش** است. این یعنی مدل بیش از حد پیچیده است و به نویز داده‌های آموزشی فیت شده، که باعث عدم توانایی آن در تعمیم به داده‌های جدید می‌شود.

همچنین می‌توان از **اعتبارسنجی متقابل** و بررسی تغییرات **R²** برای ارزیابی بهتر پیچیدگی استفاده کرد.

---

### **سوال: Contrastive Language-Image Pretraining (CLIP) به عنوان یک مثال از کاربردهای یادگیری ماشین مطرح شده است. توضیح دهید که CLIP چگونه متن و تصاویر را به هم متصل می‌کند و چه کاربردی دارد؟**

**پاسخ**:
**CLIP (Contrastive Language-Image Pretraining)** یک مدل یادگیری ماشین است که برای اتصال (Connecting) متن و تصاویر طراحی شده است. این مدل با یادگیری یک فضای مشترک (joint embedding space) برای تصاویر و متن، می‌تواند درک کند که کدام متن با کدام تصویر مطابقت دارد.

**کاربرد**: این قابلیت به CLIP اجازه می‌دهد تا وظایف مختلفی را انجام دهد، مانند:

* **جستجوی تصویری** بر اساس توضیحات متنی (مثلاً پیدا کردن "گربه‌ای با کلاه قهوه‌ای" از میان تصاویر).
* **طبقه‌بندی تصاویر** بر اساس نام کلاس‌های متنی.
* **تولید توضیحات متنی** برای تصاویر (image captioning).

CLIP می‌تواند بدون آموزش مجدد بر روی داده‌های جدید، به طور مؤثر به وظایف مختلف تعمیم پیدا کند.

