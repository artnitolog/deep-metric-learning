## [Deep Metric Learning: a (Long) Survey]((https://hav4ik.github.io/articles/deep-metric-learning-survey))
* Очень частые применения:
    - В картинках: распознавание лиц
    - В текстах: поиск по индексу
* Рассматривается конкретно Supervised Metric Learning.
* [Triplet Margin Loss](https://arxiv.org/abs/1503.03832), [Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)
* Проблемы триплета:
    - Майнинг (сложности со скейлингом)
    - Не настраивается вариативность внутри классов и между классами: не хотим крайних случаев (2 пересекающихся облака — ужасно, 2 сжатые в точку кучки — тоже ужасно)
* Quadruple Loss: сам по себе вреден, но идейно решает проблему intra/inter.
* [Lifted Structured loss](https://arxiv.org/pdf/1511.06452.pdf): вместо отдельных пар и триплетов оптимизируется расстояние во всем микробатче (подробнее в [обзоре](papers.md))