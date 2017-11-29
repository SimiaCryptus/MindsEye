### Model
This is a very simple model that performs basic logistic regression. It is expected to be trainable to about 91% accuracy on MNIST.

Code from [MnistTestBase.java:295](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L295) executed in 0.00 seconds: 
```java
    PipelineNetwork network = new PipelineNetwork();
    network.add(new BiasLayer(28, 28, 1));
    network.add(new FullyConnectedLayer(new int[]{28, 28, 1}, new int[]{10})
      .setWeights(() -> 0.001 * (Math.random() - 0.45)));
    network.add(new SoftmaxActivationLayer());
    return network;
```

Returns: 

```
    PipelineNetwork/3cc8990a-29bd-4377-9ee9-86380000006f
```



### Training
Code from [L1NormalizationTest.java:43](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/trainable/L1NormalizationTest.java#L43) executed in 180.09 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new L12Normalizer(new SampledArrayTrainable(trainingData, supervisedNetwork, 1000)) {
      @Override
      protected double getL1(NNLayer layer) {
        return 1.0;
      }
      
      @Override
      protected double getL2(NNLayer layer) {
        return 0;
      }
    };
    return new IterativeTrainer(trainable)
      .setMonitor(monitor)
      .setTimeout(3, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    Returning cached value; 2 buffers unchanged since 0.0 => 2.659568073014657
    th(0)=7.316121019648089;dx=-618951.4044870991
    Armijo: th(2.154434690031884)=85128.94157794514; dx=39495.54268673028 delta=-85121.6254569255
    Armijo: th(1.077217345015942)=42583.65794498171; dx=39495.54268673028 delta=-42576.34182396206
    Armijo: th(0.3590724483386473)=14220.135523005882; dx=39495.54268673028 delta=-14212.819401986233
    Armijo: th(0.08976811208466183)=3583.787683183067; dx=39495.84844019642 delta=-3576.471562163419
    Armijo: th(0.017953622416932366)=747.3335837401305; dx=39503.76433978621 delta=-740.0174627204824
    Armijo: th(0.002992270402822061)=156.10727002868305; dx=39656.2728836183 delta=-148.79114900903497
    Armijo: th(4.2746720040315154E-4)=52.827517054145446; dx=44690.003781673004 delta=-45.51139603449736
    Armijo: th(5.343340005039394E-5)=17.84946657963122; dx=269011.1662244525 delta=-10.533345559983129
    New Minimum: 7.316121019648089 > 5.913443641906721
    END: th(5.9370444500437714E-6)=5.913443641906721; dx=-21430.37069850576 delta=1.4026773777413686
    Iteration 1 complete. Error: 5.913443641906721 Total: 14807371813094.9630; Orientation: 0.0006; Line Search: 1.6789
    LBFGS Accumulation History: 1 points
    Returning cached value; 2 buffers unchanged since 0.0 => 2.0184154312152143
    th(0)=5.993343511009059;dx=-440141.6834279983
    Armijo: th(1.279097451943557E-5)=7.1043868932693455; dx=310975.623221564 delta=-1.111043382260286
    New Minimum: 5.993343511009059 > 5.504446469800996
    WOLF (strong): th(6.395487259717785E-6)=5.504446469800996; dx=162323.86823182862 delta=0.4888970412080633
    New Minimum: 5.504446469800996 > 5.369039672358686
    END: th(2.131829086572595E-6)=5.369039672358686; dx=-154649.8945569348 delta=0.6243038386503734
    Iteration 2 complete. Error: 5.369039672358686 Total: 14808125195791.6290; Orientation: 0.0005; Line Search: 0.5664
    LBFGS Accumulation History: 1 points
    Returning cached value; 2 buffers unchanged since 0.0 => 1.7858773
```
...[skipping 180363 bytes](etc/1.txt)...
```
    74E-7)=1.7170882267206258; dx=480.5002226704605 delta=2.1958066513994012E-4
    New Minimum: 1.7170882267206258 > 1.7170755151517787
    END: th(7.759784589049987E-8)=1.7170755151517787; dx=-522.5161884945894 delta=2.3229223398701926E-4
    Iteration 319 complete. Error: 1.7170755151517787 Total: 14983309507985.2400; Orientation: 0.0005; Line Search: 0.3659
    LBFGS Accumulation History: 1 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.48384644720501213
    th(0)=1.6441206319571096;dx=-8685.375042814736
    New Minimum: 1.6441206319571096 > 1.6437958747430774
    WOLF (strong): th(1.6717949105824098E-7)=1.6437958747430774; dx=1317.0941183090167 delta=3.247572140321342E-4
    New Minimum: 1.6437958747430774 > 1.6437078164751182
    WOLF (strong): th(8.358974552912049E-8)=1.6437078164751182; dx=473.10794501695045 delta=4.1281548199134654E-4
    END: th(2.786324850970683E-8)=1.64390717338226; dx=-6853.452149553616 delta=2.1345857484966757E-4
    Iteration 320 complete. Error: 1.6437078164751182 Total: 14984028352668.6910; Orientation: 0.0009; Line Search: 0.5352
    LBFGS Accumulation History: 1 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.5031775124878264
    th(0)=1.6826564651646292;dx=-8890.247721366746
    New Minimum: 1.6826564651646292 > 1.682410385733426
    WOLF (strong): th(6.002954916629158E-8)=1.682410385733426; dx=269.90569854298394 delta=2.460794312031389E-4
    END: th(3.001477458314579E-8)=1.6824508843321127; dx=-4230.989099456864 delta=2.055808325165387E-4
    Iteration 321 complete. Error: 1.682410385733426 Total: 14984573511275.9700; Orientation: 0.0007; Line Search: 0.3663
    LBFGS Accumulation History: 1 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.48666587743690093
    th(0)=1.649522648566244;dx=-13670.633711309041
    New Minimum: 1.649522648566244 > 1.6491524710583885
    END: th(6.466487157541656E-8)=1.6491524710583885; dx=-3691.4560347005076 delta=3.7017750785550696E-4
    Iteration 322 complete. Error: 1.6491524710583885 Total: 14984943161081.4260; Orientation: 0.0005; Line Search: 0.1832
    
```

Returns: 

```
    1.6823340663406543
```



Code from [MnistTestBase.java:141](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L141) executed in 0.01 seconds: 
```java
    PlotCanvas plot = ScatterPlot.plot(history.stream().map(step -> new double[]{step.iteration, Math.log10(step.point.getMean())}).toArray(i -> new double[i][]));
    plot.setTitle("Convergence Plot");
    plot.setAxisLabels("Iteration", "log10(Fitness)");
    plot.setSize(600, 400);
    return plot;
```

Returns: 

![Result](etc/test.1.png)



Saved model as [model0.json](etc/model0.json)

### Metrics
Code from [MnistTestBase.java:154](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L154) executed in 0.05 seconds: 
```java
    try {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      JsonUtil.writeJson(out, monitoringRoot.getMetrics());
      return out.toString();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
```

Returns: 

```
    [ "java.util.HashMap", {
      "FullyConnectedLayer/3cc8990a-29bd-4377-9ee9-863800000071" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.009528531363636355,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 5.6119149814049745E-5,
        "totalItems" : 968000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -4.74380676425864,
          "tp50" : -3.943366340699302E-4,
          "negative" : 500,
          "min" : -0.001974523594337347,
          "max" : 0.0012061841930475492,
          "tp90" : -8.023966943312149E-5,
          "mean" : 8.67695257211385E-22,
          "count" : 5000.0,
          "positive" : 4500,
          "stdDev" : 2.8848127489687934E-4,
          "tp75" : -1.628330301100068E-4,
          "zeros" : 0
        } ],
        "totalBatches" : 1936,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 0.002010785970992846,
          "tp90" : "NaN",
          "count" : 7840.0,
          "positive" : 4096,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -6.311650865938362,
          "negative" : 3744,
          "min" : -0.0015995687076433625,
          "mean" : 1.3891013498461175E-5,
          "stdDev" : 2.2717026660575083E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.01379283148238859,
          "tp50" : -2.6488100595047226,
          "negative" : 2167,
          "min" : -3.2579244531165186,
          "max" : 6.1428196916631475,
          "tp90" : -2.1937113133412867,
          "mean" : 0.5111753667914136,
          "count" : 5000.0,
          "positive" : 2833,
          "stdDev" : 2.1640557206656243,
          "tp75" : -2.3510565736014435,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "BiasLayer/3cc8990a-29bd-4377-9ee9-863800000070" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.015914380434917354,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 2.014828449793387E-5,
        "totalItems" : 968000,
        "backpropStat
```
...[skipping 791 bytes](etc/2.txt)...
```
    ative" : 413,
          "min" : -1.1009295628911401E-8,
          "mean" : -2.352415209558718E-10,
          "stdDev" : 2.2003267312846302E-9
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.960739949215013,
          "tp50" : -1.1009295628911401E-8,
          "negative" : 162449,
          "min" : -3.1361559173493446E-10,
          "max" : -3.1361559173493446E-10,
          "tp90" : -1.0233753764515209E-8,
          "mean" : 32.82857908139744,
          "count" : 392000.0,
          "positive" : 229551,
          "stdDev" : 78.08199794511191,
          "tp75" : -1.0233753764515209E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/3cc8990a-29bd-4377-9ee9-863800000072" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.0022773333336776735,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 1.30659395247934E-6,
        "totalItems" : 968000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.4918452763296735,
          "tp50" : -0.002491182202243885,
          "negative" : 500,
          "min" : -0.15700801961493768,
          "max" : 0.0,
          "tp90" : -0.0020835934237785065,
          "mean" : -4.704877771037588E-4,
          "count" : 5000.0,
          "positive" : 0,
          "stdDev" : 0.003361890745861243,
          "tp75" : -0.0021772653577804713,
          "zeros" : 4500
        } ],
        "totalBatches" : 1936,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -1.9906770118664787,
          "tp50" : 1.8478151152745716E-4,
          "negative" : 0,
          "min" : 9.012377694175151E-6,
          "max" : 0.9185834849449965,
          "tp90" : 3.800033134180522E-4,
          "mean" : 0.1,
          "count" : 5000.0,
          "positive" : 5000,
          "stdDev" : 0.22875616680823757,
          "tp75" : 3.0405413826243373E-4,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:211](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L211) executed in 0.61 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    88.7
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:218](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L218) executed in 0.06 seconds: 
```java
    try {
      TableOutput table = new TableOutput();
      MNIST.validationDataStream().map(labeledObject -> {
        try {
          int actualCategory = parse(labeledObject.label);
          double[] predictionSignal = CudaExecutionContext.gpuContexts.run(ctx -> network.eval(ctx, labeledObject.data).getData().get(0).getData());
          int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
          if (predictionList[0] == actualCategory) return null; // We will only examine mispredicted rows
          LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
          row.put("Image", log.image(labeledObject.data.toGrayImage(), labeledObject.label));
          row.put("Prediction", Arrays.stream(predictionList).limit(3)
            .mapToObj(i -> String.format("%d (%.1f%%)", i, 100.0 * predictionSignal[i]))
            .reduce((a, b) -> a + ", " + b).get());
          return row;
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
```

Returns: 

Image | Prediction
----- | ----------
![[5]](etc/test.2.png)  | 6 (55.1%), 2 (19.2%), 0 (5.5%) 
![[4]](etc/test.3.png)  | 6 (33.1%), 0 (31.8%), 4 (12.7%)
![[2]](etc/test.4.png)  | 3 (28.5%), 2 (25.3%), 8 (16.0%)
![[1]](etc/test.5.png)  | 3 (39.3%), 1 (30.4%), 9 (5.6%) 
![[3]](etc/test.6.png)  | 2 (41.8%), 3 (29.8%), 9 (10.6%)
![[6]](etc/test.7.png)  | 2 (26.6%), 6 (16.4%), 1 (14.9%)
![[9]](etc/test.8.png)  | 7 (29.7%), 9 (26.2%), 8 (22.3%)
![[2]](etc/test.9.png)  | 7 (72.7%), 2 (12.2%), 9 (7.3%) 
![[9]](etc/test.10.png) | 4 (26.2%), 9 (22.0%), 8 (21.3%)
![[7]](etc/test.11.png) | 1 (20.4%), 7 (19.7%), 9 (18.4%)




