## [Deep Metric Learning: a (Long) Survey]((https://hav4ik.github.io/articles/deep-metric-learning-survey))
***In progress: конспектирую ключевые статьи в papers.md***
* `TL;DR` большой подробный обзор методов в Supervised DML, делается акцент на SOTA-подходах, опыт в соревновании.
* Наиболее частые применения:
    - В картинках: распознавание лиц
    - В текстах: поиск по индексу
* В статье рассматривается именно Supervised Metric Learning.
* [Triplet Loss](https://arxiv.org/abs/1503.03832) — разобран в [конспектах](papers.md), [Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss).
* Проблемы триплета:
    - Майнинг (сложности со скейлингом)
    - Не настраивается вариативность внутри классов и между классами: не хотим крайних случаев (2 пересекающихся облака — ужасно, 2 сжатые в точку кучки — по-разному)
* [Quadruplet Loss](https://arxiv.org/pdf/1704.01719.pdf): сам по себе работает не очень, но идейно решает вторую проблему. Разобран в [конспектах](papers.md). В блог-посте либо намеренно используется отличный от статьи таргет, либо ошибка.

* [Lifted Structured loss](https://arxiv.org/pdf/1511.06452.pdf): вместо отдельных пар и триплетов оптимизируется расстояние во всем микробатче. Разобран в [конспектах](papers.md).

## [Contrastive Representation Learning](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html)
***In progress: конспектирую ключевые статьи в papers.md***
* `TL;DR` обзор не только методов, но и применений contrastive-подходов
