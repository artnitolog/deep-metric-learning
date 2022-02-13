# ММП ВМК: другая деятельность :)
*Все, что прямо не относится к диплому и прослушанным курсам* 👀

## Преддипломная практика `осень 2021`
* ***«Применение генеративных текстовых моделей для решения задач обработки естественного языка»***
* [Отчет по практике](materials/PracticeReport.pdf)
* [Презентация с защиты](materials/PracticeSlides.pdf)

На преддипломной практике (стажировке) я исследовал подходы к решению различных языковых задач ([Russian SuperGLUE](https://russiansuperglue.com/tasks/)) с помощью [дообучения](https://arxiv.org/pdf/2103.10385.pdf) больших [генеративных трансформеров](https://yandex.ru/lab/yalm). Основные итоги — [sota](https://russiansuperglue.com/login/submit_info/1455) на [RSG](https://russiansuperglue.com/) (сентябрь 2021: sota после human benchmkark, октябрь 2021 — 2022: sota среди подходов без использования дополнительных данных / моделей) и best practices для применения p-tuning'а.

## Спецсеминар

### Rotary Position Embeddings `08.12.21`
* [Презентация](materials/RotaryEmbeddings.pdf)
* [Статья](https://arxiv.org/abs/2104.09864), [blogpost](https://blog.eleuther.ai/rotary-embeddings/)

[Авторы](https://github.com/ZhuiyiTechnology) предлагают новый способ кодирования позиционной информации в транфсормерных моделях. Роторные эмбеддинги объединяют абсолютный и относительный подходы, легко [встраиваются](https://github.com/ZhuiyiTechnology/roformer) в разные (в том числе в attention-эффективные) архитектуры и на практике работают [лучше](https://blog.eleuther.ai/rotary-embeddings/) известных на момент рассказа подходов.

## Семинары по ММРО и практикуму (для 317/522)

### Обучение метрик `≈21.04.22`
*In progress...*

### Метрики качества классификации, задачи на площади под кривыми `08.11.2021`
* [Запись семинара:](https://youtu.be/4sKd2QElMbE)

[<img src="https://img.youtube.com/vi/4sKd2QElMbE/maxresdefault.jpg" width=400px alt="YT link">](https://youtu.be/4sKd2QElMbE)

Разбираем известные различные задачи по функционалам качества классификации ([[1](https://github.com/esokolov/ml-course-hse/blob/master/2021-fall/seminars/sem05-linclass-metrics.pdf)], [[2](https://dyakonov.org/2017/07/28/auc-roc-площадь-под-кривой-ошибок/)]).

### Декораторы `25.10.21`
* [Запись семинара:](https://youtu.be/x4yMpFjIEWM)

[<img src="https://img.youtube.com/vi/x4yMpFjIEWM/maxresdefault.jpg" width=400px alt="YT link">](https://youtu.be/x4yMpFjIEWM)

* [Ноутбук](https://github.com/mmp-practicum-team/mmp_practicum_fall_2021/blob/main/Seminars/Seminar%2009.%20Decorators/decorators_prac_2021_fall.ipynb)

Объясняю, как и зачем работать с декораторами: с самых баянистых примеров до нетривиальных конструкций. Рассматриваю способы прокидывания аргументов, избавления от лишних уровней вложенности, декораторы для декораторов. Декораторы для классов и с их помощью. Использование декораторов из стандартной библиотеки и сторонних пакетов. Различные применения, ссылки.

### LaTeX Sequel `04.10.21`
* [Запись семинара:](https://youtu.be/J3EstCmFHCs)

[<img src="https://img.youtube.com/vi/J3EstCmFHCs/mqdefault.jpg" width=400px alt="YT link">](https://youtu.be/J3EstCmFHCs)

* [Презентация, материалы](https://github.com/mmp-practicum-team/mmp_practicum_fall_2021/blob/main/Seminars/Seminar%2005.2.%20TeX%20Details/main.pdf)

Сиквел занятия по LaTeX'у с различными подробностями, тонкостями и хаками. Рассказываю, как удобно вести теховский проект, верстать презентации, сокращать объем копипасты с помощью команд, оформлять таблицы вручную. Объясняю, зачем для текстов на русском нужна особая преамбула и на что влияют кодировки. Привожу простые рецепты для вставки кода, оформления литературы по ГОСТу.