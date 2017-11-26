### Model
This is a very simple model that performs basic logistic regression. It is expected to be trainable to about 91% accuracy on MNIST.

Code from [MnistTestBase.java:272](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L272) executed in 0.00 seconds: 
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
    PipelineNetwork/e1035fb9-1fe3-4846-a360-62290000006c
```



### Training
Training a model involves a few different components. First, our model is combined mapCoords a loss function. Then we take that model and combine it mapCoords our training data to define a trainable object. Finally, we use a simple iterative scheme to refine the weights of our model. The final output is the last output value of the loss function when evaluating the last batch.

Code from [HoldoverStochasticGradientDescentTest.java:47](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/trainable/HoldoverStochasticGradientDescentTest.java#L47) executed in 300.15 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new DeltaHoldoverArrayTrainable(trainingData, supervisedNetwork, 10000);
    return new IterativeTrainer(trainable)
      .setMonitor(monitor)
      .setOrientation(new GradientDescent())
      .setTimeout(5, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    Constructing line search parameters: GD
    th(0)=2.552543750083771;dx=-430152.8359486275
    Armijo: th(2.154434690031884)=16.932289739841014; dx=5.532289136462961E-16 delta=-14.379745989757243
    Armijo: th(1.077217345015942)=16.932289739841014; dx=0.0016515976762373923 delta=-14.379745989757243
    Armijo: th(0.3590724483386473)=16.928079574666306; dx=0.023772503585363307 delta=-14.375535824582535
    Armijo: th(0.08976811208466183)=16.920826523882322; dx=0.3276459676139387 delta=-14.368282773798551
    Armijo: th(0.017953622416932366)=16.856634493554473; dx=8.263353438222481 delta=-14.304090743470702
    Armijo: th(0.002992270402822061)=16.58312549907755; dx=244.84170710621444 delta=-14.030581748993779
    Armijo: th(4.2746720040315154E-4)=14.94250067474343; dx=7587.247000736781 delta=-12.389956924659659
    Armijo: th(5.343340005039394E-5)=7.223168600942355; dx=238756.6667648951 delta=-4.670624850858584
    New Minimum: 2.552543750083771 > 1.935403612483297
    END: th(5.9370444500437714E-6)=1.935403612483297; dx=-1295.7953948433435 delta=0.6171401376004741
    Iteration 1 complete. Error: 1.935403612483297 Total: 184364604898868.2000; Orientation: 0.0006; Line Search: 13.7178
    th(0)=1.9260836200713334;dx=-481245.54429314
    Armijo: th(1.279097451943557E-5)=3.09041510282629; dx=467028.27981683606 delta=-1.1643314827549565
    New Minimum: 1.9260836200713334 > 1.8094059319164304
    WOLF (strong): th(6.395487259717785E-6)=1.8094059319164304; dx=290131.9259985918 delta=0.11667768815490298
    New Minimum: 1.8094059319164304 > 1.5871739399023923
    END: th(2.131829086572595E-6)=1.5871739399023923; dx=-152521.55696430636 delta=0.3389096801689411
    Iteration 2 complete. Error: 1.5871739399023923 Total: 184372059281790.4000; Orientation: 0.0004; Line Search: 5.9145
    th(0)=1.5844544142188954;dx=-153086.47959412122
    New Minimum: 1.5844544142188954 > 1.334011215424375
    END: th(4.592886537330983E-6)=1.334011215424375; dx=-67754.62635034008 delta=0.25044319879452037
    Iteration 3 complete. Error: 1.334011215424375 Total: 184376498647500.9400; Orientation: 0.0006; Line
```
...[skipping 18783 bytes](etc/1.txt)...
```
    lta=0.004192251780703482
    END: th(6.439034872904229E-6)=0.4243861463443579; dx=-645.2611516074734 delta=0.003625463975486354
    Iteration 45 complete. Error: 0.42381935853914077 Total: 184622918051378.1000; Orientation: 0.0004; Line Search: 4.5190
    th(0)=0.4359065201990995;dx=-1702.5691066485026
    New Minimum: 0.4359065201990995 > 0.4325836075183438
    WOLF (strong): th(1.3872480100509912E-5)=0.4325836075183438; dx=741.438200567707 delta=0.0033229126807556852
    New Minimum: 0.4325836075183438 > 0.43212575102851625
    END: th(6.936240050254956E-6)=0.43212575102851625; dx=-472.83513344114203 delta=0.0037807691705832314
    Iteration 46 complete. Error: 0.43212575102851625 Total: 184628801015876.1000; Orientation: 0.0008; Line Search: 4.4127
    th(0)=0.41792489930836263;dx=-1246.2052515726944
    New Minimum: 0.41792489930836263 > 0.41358600193269224
    WOLF (strong): th(1.4943676182657774E-5)=0.41358600193269224; dx=70.97520396188574 delta=0.004338897375670392
    END: th(7.471838091328887E-6)=0.41452625357289197; dx=-584.7557765728517 delta=0.0033986457354706623
    Iteration 47 complete. Error: 0.41358600193269224 Total: 184634660764896.2000; Orientation: 0.0004; Line Search: 4.3628
    th(0)=0.42858576271429527;dx=-1592.6427739554479
    Armijo: th(1.6097587182260573E-5)=0.4287045696425916; dx=1602.6738681694742 delta=-1.1880692829635553E-4
    New Minimum: 0.42858576271429527 > 0.42543331045983296
    WOLF (strong): th(8.048793591130286E-6)=0.42543331045983296; dx=4.306118513114658 delta=0.0031524522544623124
    END: th(2.682931197043429E-6)=0.42682110876419327; dx=-1059.8329949736979 delta=0.001764653950102002
    Iteration 48 complete. Error: 0.42543331045983296 Total: 184641920974668.3400; Orientation: 0.0006; Line Search: 5.7483
    th(0)=0.4198848649957748;dx=-1128.4422285949192
    New Minimum: 0.4198848649957748 > 0.4172405754938605
    END: th(5.78020004187913E-6)=0.4172405754938605; dx=-705.1543994540168 delta=0.0026442895019143098
    Iteration 49 complete. Error: 0.4172405754938605 Total: 184646372474759.1000; Orientation: 0.0005; Line Search: 2.9423
    
```

Returns: 

```
    0.42251874317029753
```



Code from [MnistTestBase.java:131](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L131) executed in 0.01 seconds: 
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
Code from [MnistTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L144) executed in 0.70 seconds: 
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
      "FullyConnectedLayer/e1035fb9-1fe3-4846-a360-62290000006e" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.011840994872058808,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 8.229890200882356E-6,
        "totalItems" : 2040000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.228915406266853,
          "tp50" : -2.4140403097847096E-5,
          "negative" : 5000,
          "min" : -1.996638250023719E-4,
          "max" : 1.7973362716299285E-4,
          "tp90" : -2.4507238572215382E-6,
          "mean" : 2.938454535406963E-23,
          "count" : 50000.0,
          "positive" : 45000,
          "stdDev" : 2.727508727043206E-5,
          "tp75" : -7.75461260144193E-6,
          "zeros" : 0
        } ],
        "totalBatches" : 408,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 0.002068222132014809,
          "tp90" : "NaN",
          "count" : 7840.0,
          "positive" : 4244,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -3.6821939932922696,
          "negative" : 3596,
          "min" : -0.0014368788817186963,
          "mean" : 4.8622614313991755E-5,
          "stdDev" : 3.718082951637612E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.1513839205206089,
          "tp50" : -3.143308172700417,
          "negative" : 18592,
          "min" : -5.723694722448631,
          "max" : 8.921608654173145,
          "tp90" : -2.4473028489408786,
          "mean" : 1.0435153083651003,
          "count" : 50000.0,
          "positive" : 31408,
          "stdDev" : 2.84839338259271,
          "tp75" : -2.6683875751822708,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/e1035fb9-1fe3-4846-a360-62290000006f" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.003196548265686279,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 1.6494448745098042E-7,
        "totalItems" : 2040000,
     
```
...[skipping 786 bytes](etc/2.txt)...
```
    2641643,
          "tp90" : 4.357739357759038E-5,
          "mean" : 0.1,
          "count" : 50000.0,
          "positive" : 50000,
          "stdDev" : 0.24690250994110483,
          "tp75" : 3.0168142128200437E-5,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "BiasLayer/e1035fb9-1fe3-4846-a360-62290000006d" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.017772764743137257,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 4.4143586796078395E-6,
        "totalItems" : 2040000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -8.329679493546829,
          "tp50" : -2.1187671925954548E-7,
          "negative" : 1935062,
          "min" : -1.49203602809808E-7,
          "max" : 1.5170233961621298E-7,
          "tp90" : -1.881802715484711E-7,
          "mean" : 1.490247716482242E-11,
          "count" : 3920000.0,
          "positive" : 1984938,
          "stdDev" : 3.152185117097562E-8,
          "tp75" : -1.9536768484575435E-7,
          "zeros" : 0
        } ],
        "totalBatches" : 408,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 6.686985321750085E-9,
          "tp90" : "NaN",
          "count" : 784.0,
          "positive" : 382,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -8.956189387801887,
          "negative" : 402,
          "min" : -6.5216659481195E-9,
          "mean" : -1.086661688973945E-10,
          "stdDev" : 2.1005852900516152E-9
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.841498049043477,
          "tp50" : -6.5216659481195E-9,
          "negative" : 1588197,
          "min" : 4.180311129152315E-9,
          "max" : 4.180311129152315E-9,
          "tp90" : -6.0636373469097615E-9,
          "mean" : 33.450025254991715,
          "count" : 3920000.0,
          "positive" : 2331803,
          "stdDev" : 78.72492397295144,
          "tp75" : -6.5216659481195E-9,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:201](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L201) executed in 0.94 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    89.49000000000001
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:208](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L208) executed in 0.07 seconds: 
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
![[5]](etc/test.2.png)  | 6 (45.8%), 2 (32.1%), 4 (6.6%) 
![[4]](etc/test.3.png)  | 0 (59.3%), 6 (21.3%), 2 (5.8%) 
![[1]](etc/test.4.png)  | 3 (36.8%), 1 (18.6%), 5 (12.7%)
![[3]](etc/test.5.png)  | 2 (48.2%), 3 (32.1%), 8 (7.9%) 
![[6]](etc/test.6.png)  | 2 (31.8%), 6 (27.1%), 7 (14.4%)
![[2]](etc/test.7.png)  | 7 (81.8%), 2 (7.8%), 9 (7.3%)  
![[7]](etc/test.8.png)  | 9 (50.1%), 7 (37.8%), 4 (6.2%) 
![[9]](etc/test.9.png)  | 8 (27.6%), 9 (25.1%), 4 (21.1%)
![[7]](etc/test.10.png) | 1 (62.3%), 7 (18.8%), 9 (7.1%) 
![[7]](etc/test.11.png) | 4 (41.7%), 9 (28.6%), 7 (21.2%)




