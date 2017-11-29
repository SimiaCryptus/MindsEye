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
    PipelineNetwork/3cc8990a-29bd-4377-9ee9-863800000037
```



### Training
Code from [OWLQNTest.java:42](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/orient/OWLQNTest.java#L42) executed in 301.38 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 10000);
    return new IterativeTrainer(trainable)
      .setIterationsPerSample(100)
      .setMonitor(monitor)
      .setOrientation(new ValidatingOrientationWrapper(new OwlQn()))
      .setTimeout(5, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: OWL/QN
    Returning cached value; 2 buffers unchanged since 0.0 => 2.3847031386732453
    -343541.14487505634 vs (-343084.18136640533, -343805.51931744604); probe=0.001
    -343216.4487200453 vs (-343084.18136640533, -343156.3320959327); probe=1.0E-4
    -343180.72589866904 vs (-343084.18136640533, -343084.9028913498); probe=1.0E-6
    th(0)=2.3847031386732453;dx=-343084.18136640533
    -7.667965859086564E-6 vs (-1.480593996381028E-8, 0.0); probe=0.001
    -7.667965859086564E-5 vs (-1.480593996381028E-8, 0.0); probe=1.0E-4
    -0.007667965859086565 vs (-1.480593996381028E-8, 0.0); probe=1.0E-6
    Armijo: th(2.154434690031884)=18.13976536408881; dx=-1.480593996381028E-8 delta=-15.755062225415564
    8.000939748713519E-5 vs (0.004597698035277361, 3.510406523062451E-78); probe=0.001
    8.000939748713518E-4 vs (0.004597698035277361, 2.229282962129354E-9); probe=1.0E-4
    0.0045975112764481876 vs (0.004597698035277361, 0.004597653809837924); probe=1.0E-6
    Armijo: th(1.077217345015942)=17.60341721731044; dx=0.004597698035277361 delta=-15.218714078637195
    0.005410239686377624 vs (0.02753217513114715, 2.502974129717051E-7); probe=0.001
    0.011983205917601311 vs (0.02753217513114715, 0.01165651227998482); probe=1.0E-4
    0.027532200488611727 vs (0.02753217513114715, 0.02753222564026847); probe=1.0E-6
    Armijo: th(0.3590724483386473)=17.600264993955474; dx=0.02753217513114715 delta=-15.215561855282228
    0.20613369532831233 vs (0.4727282617858429, 0.11803338501963244); probe=0.001
    0.40194938429798543 vs (0.4727282617858429, 0.397443168420022); probe=1.0E-4
    0.4157284815616726 vs (0.4727282617858429, 0.47142535164902555); probe=1.0E-6
    Armijo: th(0.08976811208466183)=17.588782431959256; dx=0.4727282617858429 delta=-15.20407929328601
    10.015829031601484 vs (13.279379101591484, 9.239689648928994); probe=0.001
    11.623401757340215 vs (13.279379101591484, 12.663139150109508); probe=1.0E-4
    12.20981077481408 vs (13.279379101591484, 13.274341040874516); probe=1.0E-6
    Armijo: th(0.0179536224169
```
...[skipping 33526 bytes](etc/1.txt)...
```
    771700697656 vs (-577.5906096253096, -578.2907132239168); probe=1.0E-6
    New Minimum: 0.49865379884251465 > 0.4932320251660275
    END: th(9.613427552967019E-6)=0.4932320251660275; dx=-577.5906096253096 delta=0.005421773676487163
    Iteration 27 complete. Error: 0.4932320251660275 Total: 13624563561464.3000; Orientation: 0.0011; Line Search: 6.8631
    Orientation vanished. Popping history element from 0.5107680914351721, 0.5043097085763621, 0.49865379884251465, 0.4932320251660275
    LBFGS Accumulation History: 3 points
    -2574.5587247910953 vs (-3133.979921862451, -2638.230969103866); probe=0.001
    -2520.3377529405575 vs (-3133.979921862451, -2529.8612974106563); probe=1.0E-4
    -2514.37546774128 vs (-3133.979921862451, -2517.9379933674472); probe=1.0E-6
    th(0)=0.4932320251660275;dx=-3133.979921862451
    3471.942720146575 vs (5439.773286871076, 5478.0658610215505); probe=0.001
    -20026.665500351686 vs (5439.773286871076, 5405.127721900621); probe=1.0E-4
    -2537909.5082101864 vs (5439.773286871076, 5397.939263806435); probe=1.0E-6
    Armijo: th(2.071150181022047E-5)=0.5083480764947333; dx=5439.773286871076 delta=-0.01511605132870586
    1557.5981185375267 vs (1422.4662414824732, 1684.7343659723826); probe=0.001
    1439.5720599360304 vs (1422.4662414824732, 1448.6811141886715); probe=1.0E-4
    1426.5935740532532 vs (1422.4662414824732, 1422.7283773449487); probe=1.0E-6
    New Minimum: 0.4932320251660275 > 0.4904676822148395
    WOLF (strong): th(1.0355750905110236E-5)=0.4904676822148395; dx=1422.4662414824732 delta=0.0027643429511879924
    -1342.4452520142509 vs (-1190.9004479381617, -1501.1320363049838); probe=0.001
    -1202.8079550400284 vs (-1190.9004479381617, -1221.9362951885723); probe=1.0E-4
    -1187.443194678286 vs (-1190.9004479381617, -1191.2108201247488); probe=1.0E-6
    New Minimum: 0.4904676822148395 > 0.4900607836533293
    END: th(3.4519169683700785E-6)=0.4900607836533293; dx=-1190.9004479381617 delta=0.0031712415126982085
    Iteration 28 complete. Error: 0.4900607836533293 Total: 13638231461038.4360; Orientation: 0.0011; Line Search: 13.6667
    
```

Returns: 

```
    0.4900607836533293
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
Code from [MnistTestBase.java:154](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L154) executed in 0.54 seconds: 
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
      "BiasLayer/3cc8990a-29bd-4377-9ee9-863800000038" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.012438506483002829,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 1.9978107230594894E-6,
        "totalItems" : 3530000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -8.160654591470468,
          "tp50" : -1.9953567183832054E-7,
          "negative" : 1967060,
          "min" : -1.6442609118426604E-7,
          "max" : 1.3965667562558325E-7,
          "tp90" : -1.778485974450442E-7,
          "mean" : 8.004072584931535E-12,
          "count" : 3920000.0,
          "positive" : 1952940,
          "stdDev" : 3.33656345376923E-8,
          "tp75" : -1.8432742715046232E-7,
          "zeros" : 0
        } ],
        "totalBatches" : 706,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 0.0,
          "tp90" : "NaN",
          "count" : 784.0,
          "positive" : 0,
          "tp75" : "NaN",
          "zeros" : 784,
          "meanExponent" : "NaN",
          "negative" : 0,
          "min" : 0.0,
          "mean" : 0.0,
          "stdDev" : 0.0
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 2.112317628155531,
          "tp50" : 0.0,
          "negative" : 0,
          "min" : 0.0,
          "max" : 0.0,
          "tp90" : 0.0,
          "mean" : 33.4151068877551,
          "count" : 3920000.0,
          "positive" : 751747,
          "stdDev" : 78.66960947789427,
          "tp75" : 0.0,
          "zeros" : 3168253
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "FullyConnectedLayer/3cc8990a-29bd-4377-9ee9-863800000039" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.007208563311331441,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 5.35706889439094E-6,
        "totalItems" : 3530000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -5.912808983638872,
          "tp50" : -3.8282682760734746E-5,
          "negative" : 5000,
          "min" : -1.
```
...[skipping 614 bytes](etc/2.txt)...
```
    
          "negative" : 2991,
          "min" : -0.001583714374759889,
          "mean" : 6.526737959799504E-5,
          "stdDev" : 3.608688425646005E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.23274917345926716,
          "tp50" : -1.345766902256856,
          "negative" : 8423,
          "min" : -4.1312165182466805,
          "max" : 10.19217983309459,
          "tp90" : -0.7480244632539115,
          "mean" : 2.288385499287399,
          "count" : 50000.0,
          "positive" : 41577,
          "stdDev" : 2.547839134017002,
          "tp75" : -0.9443880503739662,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/3cc8990a-29bd-4377-9ee9-86380000003a" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.0019900946079320133,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 9.187111966005664E-8,
        "totalItems" : 3530000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -3.4822798052990582,
          "tp50" : -2.4734518654436307E-4,
          "negative" : 5000,
          "min" : -0.09081669254219733,
          "max" : 0.0,
          "tp90" : -2.055143420924845E-4,
          "mean" : -6.804696852938784E-5,
          "count" : 50000.0,
          "positive" : 0,
          "stdDev" : 0.0013791001469162921,
          "tp75" : -2.1497146395027642E-4,
          "zeros" : 45000
        } ],
        "totalBatches" : 706,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.155084776089597,
          "tp50" : 5.5708275424593766E-5,
          "negative" : 0,
          "min" : 5.728096552790493E-7,
          "max" : 0.9630877876117298,
          "tp90" : 1.6291778347796612E-4,
          "mean" : 0.1,
          "count" : 50000.0,
          "positive" : 50000,
          "stdDev" : 0.2327919009520555,
          "tp75" : 1.178441660978841E-4,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:211](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L211) executed in 0.51 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    87.87
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:218](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L218) executed in 0.19 seconds: 
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
![[5]](etc/test.2.png)  | 2 (37.4%), 6 (31.6%), 4 (11.2%)
![[4]](etc/test.3.png)  | 0 (40.5%), 6 (32.0%), 2 (10.7%)
![[5]](etc/test.4.png)  | 3 (46.8%), 5 (40.6%), 8 (7.9%) 
![[1]](etc/test.5.png)  | 3 (42.7%), 5 (15.1%), 1 (12.2%)
![[9]](etc/test.6.png)  | 4 (37.5%), 9 (34.6%), 8 (8.3%) 
![[3]](etc/test.7.png)  | 2 (35.8%), 3 (35.0%), 9 (10.3%)
![[9]](etc/test.8.png)  | 7 (31.8%), 9 (29.8%), 8 (19.7%)
![[2]](etc/test.9.png)  | 7 (71.3%), 9 (10.3%), 2 (9.4%) 
![[7]](etc/test.10.png) | 9 (64.5%), 7 (25.2%), 4 (5.2%) 
![[9]](etc/test.11.png) | 8 (27.6%), 9 (27.2%), 4 (19.6%)




