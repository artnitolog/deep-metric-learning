## Deep Metric Learning via Lifted Structured Feature Embedding
* **[arxiv 1511.06452](https://arxiv.org/pdf/1511.06452.pdf), [github](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16)**
* Рассматривают следующе развитие лоссов: Contrastive Embedding -> Triplet Embedding -> *Lifted Structured Embedding*
img[src$="centerme"] {
  display:block;
  margin: 0 auto;
}

<p style="text-align:center;"><img src="vis/lse_loss.png" width="300"></p>

* Страшная формула предложенного лосса с простой интерпретацией: строим связи внутри батча (как всегда: притягивая объекты одного класса и отдяляя объекты разных)
* Другое объяснение:
    1. Берем пару из одного класса, сближаем 