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
    PipelineNetwork/3cc8990a-29bd-4377-9ee9-863800000040
```



### Training
Code from [QQNTest.java:44](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/orient/QQNTest.java#L44) executed in 308.11 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    //return new IterativeTrainer(new SampledArrayTrainable(trainingData, supervisedNetwork, 10000))
    return new ValidatingTrainer(
      new SampledArrayTrainable(trainingData, supervisedNetwork, 1000, 10000),
      new ArrayTrainable(trainingData, supervisedNetwork)
    )
      .setMonitor(monitor)
      .setOrientation(new QQN())
      .setTimeout(5, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    Epoch parameters: 1000, 1
    Phase 0: TrainingPhase{trainingSubject=SampledCachedTrainable{inner=PerformanceWrapper{inner=SampledArrayTrainable{inner=ArrayTrainable{inner=com.simiacryptus.mindseye.eval.GpuTrainable@66f73d3d}}}}, orientation=com.simiacryptus.mindseye.opt.orient.QQN@2c427287}
    resetAndMeasure; trainingSize=1000
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD
    Returning cached value; 2 buffers unchanged since 0.0 => 2.3837684318286714
    th(0)=2.3837684318286714;dx=-340180.5286730842
    Armijo: th(2.154434690031884)=16.13651633170227; dx=0.0 delta=-13.7527478998736
    Armijo: th(1.077217345015942)=16.13651633170227; dx=0.0 delta=-13.7527478998736
    Armijo: th(0.3590724483386473)=16.13651633170227; dx=4.159622254917866E-149 delta=-13.7527478998736
    Armijo: th(0.08976811208466183)=16.13651633170227; dx=1.1348812985406095E-28 delta=-13.7527478998736
    Armijo: th(0.017953622416932366)=16.12584746329997; dx=5.938702697193425 delta=-13.742079031471299
    Armijo: th(0.002992270402822061)=15.851475437708881; dx=154.75608041237487 delta=-13.46770700588021
    Armijo: th(4.2746720040315154E-4)=14.745552648930868; dx=6580.830843339283 delta=-12.361784217102198
    Armijo: th(5.343340005039394E-5)=5.971456509820708; dx=215535.86738106154 delta=-3.587688077992037
    New Minimum: 2.3837684318286714 > 1.8038325651997682
    END: th(5.9370444500437714E-6)=1.8038325651997682; dx=-53469.9958553448 delta=0.5799358666289032
    Returning cached value; 2 buffers unchanged since 0.0 => 1.8038325651997682
    Overall network state change: {FullyConnectedLayer=0.9949753877206098, BiasLayer=0.0}
    Iteration 1 complete. Error: 1.8038325651997682 (1000 in 1.526 seconds; 0.001 in orientation, 0.196 in gc, 1.524 in line search; 1.905 eval time)
    Epoch 1 result with 2 iterations, 1000/1000000 samples: {validation *= 2^-0.36660; training *= 2^-0.402; Overtraining = 1.10}, {itr*=0.64, len*=0.74} 0 since improvement; 10.8616 validation time
    Epoch parameters: 740, 1
    Phase 0: TrainingPhase{trainingSubject=SampledCachedTrainable{in
```
...[skipping 180826 bytes](etc/1.txt)...
```
    ached value; 2 buffers unchanged since 0.0 => 0.2694334432417048
    Overall network state change: {FullyConnectedLayer=0.9959913267798868, BiasLayer=0.9807151696813378}
    Iteration 175 complete. Error: 0.2694334432417048 (8012 in 2.397 seconds; 0.001 in orientation, 0.340 in gc, 2.395 in line search; 2.391 eval time)
    Orientation vanished. Popping history element from 0.271995052792559, 0.27156935918634895, 0.27130074566488316, 0.2709724639523646, 0.27065102955494247, 0.2694334432417048
    Orientation vanished. Popping history element from 0.271995052792559, 0.27156935918634895, 0.27130074566488316, 0.2709724639523646, 0.27065102955494247
    Orientation vanished. Popping history element from 0.271995052792559, 0.27156935918634895, 0.27130074566488316, 0.2709724639523646
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 6.966695499720704E-5 => 0.2694334432417048
    th(0)=0.2694334432417048;dx=-98.77932340027363
    Armijo: th(1.5009290459487294E-4)=0.30240930773322383; dx=1025.8975101023054 delta=-0.032975864491519014
    Armijo: th(7.504645229743647E-5)=0.27542553375731327; dx=423.76543385460985 delta=-0.00599209051560845
    New Minimum: 0.2694334432417048 > 0.2692661860838495
    WOLF (strong): th(2.501548409914549E-5)=0.2692661860838495; dx=69.26232844812738 delta=1.6725715785531614E-4
    New Minimum: 0.2692661860838495 > 0.26919485017795974
    END: th(6.253871024786372E-6)=0.26919485017795974; dx=-57.20959317294325 delta=2.3859306374507394E-4
    Returning cached value; 2 buffers unchanged since 0.0 => 0.26919485017795974
    Overall network state change: {FullyConnectedLayer=0.9996428631138806, BiasLayer=0.9977491869162364}
    Iteration 176 complete. Error: 0.26919485017795974 (8012 in 3.210 seconds; 0.002 in orientation, 0.466 in gc, 3.208 in line search; 3.204 eval time)
    Epoch 21 result with 42 iterations, 8012/1000000 samples: {validation *= 2^-0.05001; training *= 2^-0.257; Overtraining = 5.14}, {itr*=1.74, len*=1.60} 0 since improvement; 5.7572 validation time
    Training timeout
    Training halted
    
```

Returns: 

```
    0.3113597718585871
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
Code from [MnistTestBase.java:154](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L154) executed in 3.73 seconds: 
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
      "BiasLayer/3cc8990a-29bd-4377-9ee9-863800000041" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.014283087975664086,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 3.060167299155664E-6,
        "totalItems" : 2900734,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -9.576022883790728,
          "tp50" : -5.898348028870352E-8,
          "negative" : 11474699,
          "min" : -3.187131349143358E-8,
          "max" : 3.149328200345575E-8,
          "tp90" : -5.099356788754482E-8,
          "mean" : -2.140874977111443E-12,
          "count" : 2.352E7,
          "positive" : 12045301,
          "stdDev" : 6.665927304968679E-9,
          "tp75" : -5.338044891352089E-8,
          "zeros" : 0
        } ],
        "totalBatches" : 888,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 3.735826847232356E-8,
          "tp90" : "NaN",
          "count" : 784.0,
          "positive" : 412,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -8.173994485940774,
          "negative" : 372,
          "min" : -5.8379462332573105E-8,
          "mean" : -1.6632330242406767E-10,
          "stdDev" : 1.3000023420473623E-8
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.2157523550059794,
          "tp50" : -5.8379462332573105E-8,
          "negative" : 8785617,
          "min" : 9.507867149570997E-9,
          "max" : 9.507867149570997E-9,
          "tp90" : -5.541728074769429E-8,
          "mean" : 33.585364880797364,
          "count" : 2.352E7,
          "positive" : 14734383,
          "stdDev" : 78.91964738880971,
          "tp75" : -5.8379462332573105E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "FullyConnectedLayer/3cc8990a-29bd-4377-9ee9-863800000042" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.009006548647342367,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 7.555070639876802E-6,
        "totalItems" : 2900734,
```
...[skipping 791 bytes](etc/2.txt)...
```
    ,
          "negative" : 3610,
          "min" : -0.002703291993270593,
          "mean" : 5.359132279084486E-5,
          "stdDev" : 5.554513817153346E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.37596392248783506,
          "tp50" : -6.11419172707893,
          "negative" : 115388,
          "min" : -10.899923263298074,
          "max" : 14.873947035607486,
          "tp90" : -4.62295257472355,
          "mean" : 1.3580000525416587,
          "count" : 300000.0,
          "positive" : 184612,
          "stdDev" : 4.69592827389279,
          "tp75" : -5.101705276334706,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/3cc8990a-29bd-4377-9ee9-863800000043" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.0019883731607241465,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 2.0555833953803825E-7,
        "totalItems" : 2900734,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -4.343107439802461,
          "tp50" : -3.444087907313746E-5,
          "negative" : 30000,
          "min" : -318.57072957947753,
          "max" : 0.0,
          "tp90" : -3.33692401119442E-5,
          "mean" : -0.06021158867510583,
          "count" : 300000.0,
          "positive" : 0,
          "stdDev" : 28.564578446925594,
          "tp75" : -3.35570480092452E-5,
          "zeros" : 270000
        } ],
        "totalBatches" : 888,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -3.66960507018649,
          "tp50" : 1.2005439506629644E-8,
          "negative" : 0,
          "min" : 9.781133095772578E-15,
          "max" : 0.9992240776630705,
          "tp90" : 1.2649550568101963E-7,
          "mean" : 0.1,
          "count" : 300000.0,
          "positive" : 300000,
          "stdDev" : 0.27148301328812574,
          "tp75" : 5.985387080851868E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:211](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L211) executed in 0.69 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    91.42
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
![[5]](etc/test.2.png)  | 6 (96.6%), 2 (1.6%), 4 (0.8%)  
![[4]](etc/test.3.png)  | 0 (74.0%), 6 (18.8%), 5 (4.9%) 
![[2]](etc/test.4.png)  | 3 (48.5%), 2 (35.9%), 8 (6.2%) 
![[3]](etc/test.5.png)  | 2 (61.5%), 3 (36.4%), 8 (1.6%) 
![[9]](etc/test.6.png)  | 4 (55.2%), 9 (24.8%), 8 (10.4%)
![[7]](etc/test.7.png)  | 4 (78.1%), 9 (17.3%), 7 (2.6%) 
![[2]](etc/test.8.png)  | 9 (70.3%), 3 (10.8%), 4 (6.9%) 
![[9]](etc/test.9.png)  | 8 (45.3%), 9 (31.8%), 2 (10.4%)
![[9]](etc/test.10.png) | 3 (33.3%), 9 (30.4%), 4 (26.1%)
![[3]](etc/test.11.png) | 8 (38.4%), 3 (34.7%), 5 (12.6%)




