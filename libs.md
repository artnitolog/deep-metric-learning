## [Pytorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/)
* [docs](https://kevinmusgrave.github.io/pytorch-metric-learning/), [github](https://github.com/KevinMusgrave/pytorch-metric-learning)
* Один из самых популярных репозиториев для работы с DML: реализованы основные лоссы, метрики, инструменты для майнинга, обучения и замеров.
* Полностью написан на питоне и интегрирован с торчом, то есть максимально удобен для исследований.

## [tensorflow / similarity](https://github.com/tensorflow/similarity)
* [github](https://github.com/tensorflow/similarity), [docs](https://github.com/tensorflow/similarity/tree/master/api)
* [Свежая](https://blog.tensorflow.org/2021/09/introducing-tensorflow-similarity.html) библиотека для обучения на задачи DML поверх TF. Интересна скорее с точки зрения реализованных методов и замеров, нежели использования.
* Есть [примеры](https://github.com/tensorflow/similarity/tree/master/examples).

## [Setntence Transformers](https://www.sbert.net/)
* [docs](https://www.sbert.net/), [github](https://github.com/UKPLab/sentence-transformers)
* Собраны предобученные модели для эмбеддингов текстов и изображений, а также инструменты для файнтюнинга на новые задачи.
* Преимущественно для трансформерных эмбеддингов текстов, написано поверх торча и huggingface.
* Есть подробные [примеры](https://www.sbert.net/docs/quickstart.html).

## [Faiss](https://faiss.ai/)
* [docs](https://faiss.ai/), [github](https://github.com/facebookresearch/faiss)
* Библиотека от фейсбука для основных сценариев инференса с обученными эмбеддингами.
* Есть [примеры](https://github.com/facebookresearch/faiss/wiki/Getting-started) и подробная [вики](https://github.com/facebookresearch/faiss/wiki/FAQ).
* Написана на C++/CUDA, можно использовать на CPU/GPU, есть обертка на питоне.

## [Qdrant](https://qdrant.tech/)
* [docs](https://qdrant.tech/documentation/), [github](https://github.com/qdrant/qdrant)
* Движок для быстрого поиска по эмбеддингам, а также их хранению и обновлению.
* Есть [примеры](https://blog.qdrant.tech/neural-search-tutorial-3f034ab13adc), [демки](https://demo.qdrant.tech/), [комьюнити](https://github.com/qdrant/qdrant#contacts).
* Движок использовался на [треке ODS](https://ods.ai/tracks/open-science-soc2021/competitions/metric-learning-hack-soc2021).
* Написан на Rust, запускается с докером, применяется через REST API, есть обертка на питоне.
* Есть полезный [репозиторий](https://github.com/qdrant/awesome-metric-learning).

## [Embedding Projector](https://projector.tensorflow.org/)
* Удобная тулза для визуализации эмбеддингов.

## Больше не обновляются, но могут пригодиться
*(Есть множество репозиториев к конкретным статьям — здесь только общие или большие проекты)*
* [MatchZoo](https://github.com/NTMC-Community/MatchZoo): преимущественно QD-модели, реализации на TF и Keras
* [bnu-wangxun / Deep_Metric](https://github.com/bnu-wangxun/Deep_Metric): реализованы релевантные на '19-20 методы, все поверх PyTorch, впоследствии [статья](https://arxiv.org/pdf/1912.06798.pdf) с новым методом.
* [Confusezius / Deep-Metric-Learning-Baselines](https://github.com/Confusezius/Deep-Metric-Learning-Baselines): основные методы DML ('19-20) по итогу авторы написали [статью](https://arxiv.org/abs/2002.08473) и сделали еще один [репозиторий](https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch).