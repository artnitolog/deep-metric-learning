## Deep Metric Learning via Lifted Structured Feature Embedding
* **[arxiv 1511.06452](https://arxiv.org/pdf/1511.06452.pdf), [github](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16)**

<p align="center"><img src="vis/lse_loss.png" width="500"></p>

* Страшная формула предложенного лосса с простой интерпретацией: строим связи внутри целого батча (как всегда: притягивая объекты одного класса и отдяляя объекты разных) — это следующий шаг после contrastive pair и триплета.
* Как читать лосс:
    1. Берем пару из одного класса `(i, j)` — сближаем
    2. Рассматриваем `i` в качестве anchor'а и ищем в других классах **самый** hard negative
    3. То же самое проделываем для `j`
    4. Из 2. и 3. выбираем более сложный hard negative — отдаляем
* Какие беды:
    1. Нет гладкости
    2. Переусложненный майнинг
* Решение классическое (делаем upper bound):

* Рассматривают следующе развитие лоссов: Contrastive Embedding -> Triplet Embedding -> *Lifted Structured Embedding*
* *Extreme classification* (наконец нашлелся термин)
* Преимущество лосса — утилизируется весь *микробатч* (возможно, одни из первых).
* Собрали датасет: [Stanford Online Products Dataset](https://cvgl.stanford.edu/projects/lifted_struct/) — 120k картинок, 23k классов — впоследствии использовался для сравнения различных ML-подходов
* 