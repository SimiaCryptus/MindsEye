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
    PipelineNetwork/e1035fb9-1fe3-4846-a360-62290000005a
```



### Training
Code from [SingleOrthantTrustRegionTest.java:43](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/region/SingleOrthantTrustRegionTest.java#L43) executed in 182.56 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 10000);
    TrustRegionStrategy trustRegionStrategy = new TrustRegionStrategy() {
      @Override
      public TrustRegion getRegionPolicy(NNLayer layer) {
        return new SingleOrthant();
      }
    };
    return new IterativeTrainer(trainable)
      .setIterationsPerSample(100)
      .setMonitor(monitor)
      //.setOrientation(new ValidatingOrientationWrapper(trustRegionStrategy))
      .setOrientation(trustRegionStrategy)
      .setTimeout(3, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    LBFGS Accumulation History: 1 points
    Constructing line search parameters: GD+Trust
    Returning cached value; 2 buffers unchanged since 0.0 => 2.4700467805829587
    th(0)=2.4700467805829587;dx=-375801.0196667463
    Armijo: th(2.154434690031884)=18.198525974644355; dx=0.003187157752985642 delta=-15.728479194061396
    Armijo: th(1.077217345015942)=18.196781484225568; dx=0.0031440184629170664 delta=-15.72673470364261
    Armijo: th(0.3590724483386473)=18.19446673538653; dx=0.044844242051243495 delta=-15.724419954803572
    Armijo: th(0.08976811208466183)=18.178271762896465; dx=0.2380818074503904 delta=-15.708224982313507
    Armijo: th(0.017953622416932366)=18.132019464023; dx=8.547619785732511 delta=-15.661972683440041
    Armijo: th(0.002992270402822061)=17.804115251803868; dx=290.2930183551246 delta=-15.33406847122091
    Armijo: th(4.2746720040315154E-4)=15.243955172664144; dx=15151.796956438435 delta=-12.773908392081186
    Armijo: th(5.343340005039394E-5)=4.168506554729793; dx=145211.2626354342 delta=-1.698459774146834
    New Minimum: 2.4700467805829587 > 1.9264541653653326
    END: th(5.9370444500437714E-6)=1.9264541653653326; dx=-107273.7484049417 delta=0.543592615217626
    Iteration 1 complete. Error: 1.9264541653653326 Total: 183993540606401.4700; Orientation: 0.0010; Line Search: 10.9627
    LBFGS Accumulation History: 2 points
    Returning cached value; 2 buffers unchanged since 0.0 => 1.9264541653653326
    th(0)=1.9264541653653326;dx=-242746.06355580126
    New Minimum: 1.9264541653653326 > 1.6628763902591972
    WOLF (strong): th(1.279097451943557E-5)=1.6628763902591972; dx=100154.35037828339 delta=0.26357777510613545
    New Minimum: 1.6628763902591972 > 1.5242671440759414
    END: th(6.395487259717785E-6)=1.5242671440759414; dx=-30426.46774504904 delta=0.4021870212893912
    Iteration 2 complete. Error: 1.5242671440759414 Total: 183996141346818.4700; Orientation: 0.0007; Line Search: 2.5999
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 0.0 => 1.5242671440759414
    th(0)=1.5242671440759414;dx=-282645.6639090522
    A
```
...[skipping 43712 bytes](etc/1.txt)...
```
    8 complete. Error: 0.3643169936873617 Total: 184155512632181.4000; Orientation: 0.0020; Line Search: 3.9241
    Orientation vanished. Popping history element from 0.3697124634739601, 0.3671828213273258, 0.3660391480017203, 0.3643169936873617
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3643169936873617
    th(0)=0.3643169936873617;dx=-581.3804962986078
    New Minimum: 0.3643169936873617 > 0.36346425773632407
    END: th(5.990054863083504E-6)=0.36346425773632407; dx=-148.0968886024 delta=8.527359510376309E-4
    Iteration 69 complete. Error: 0.36346425773632407 Total: 184156850941708.5000; Orientation: 0.0021; Line Search: 1.3360
    Orientation vanished. Popping history element from 0.3671828213273258, 0.3660391480017203, 0.3643169936873617, 0.36346425773632407
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.36346425773632407
    th(0)=0.36346425773632407;dx=-371.8835217257461
    New Minimum: 0.36346425773632407 > 0.36213797243273405
    END: th(1.2905181992221287E-5)=0.36213797243273405; dx=-165.0360710669367 delta=0.0013262853035900246
    Iteration 70 complete. Error: 0.36213797243273405 Total: 184158133891906.1600; Orientation: 0.0011; Line Search: 1.2818
    Orientation vanished. Popping history element from 0.3660391480017203, 0.3643169936873617, 0.36346425773632407, 0.36213797243273405
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.36213797243273405
    th(0)=0.36213797243273405;dx=-457.7043510053048
    Armijo: th(2.7803371765216317E-5)=0.36219887842609083; dx=322.04256875001266 delta=-6.0905993356785704E-5
    New Minimum: 0.36213797243273405 > 0.3610628525154635
    WOLF (strong): th(1.3901685882608158E-5)=0.3610628525154635; dx=4.372401314140022 delta=0.0010751199172705506
    END: th(4.6338952942027194E-6)=0.36153322396688237; dx=-207.95722513429607 delta=6.047484658516789E-4
    Iteration 71 complete. Error: 0.3610628525154635 Total: 184162087500271.8800; Orientation: 0.0016; Line Search: 3.9519
    
```

Returns: 

```
    0.3610628525154635
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
Code from [MnistTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L144) executed in 0.64 seconds: 
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
      "FullyConnectedLayer/e1035fb9-1fe3-4846-a360-62290000005c" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.012266327694285721,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 7.5318380709999985E-6,
        "totalItems" : 1400000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.475041567556407,
          "tp50" : -1.8251136889611776E-5,
          "negative" : 5000,
          "min" : -1.999858845213518E-4,
          "max" : 1.908482190583407E-4,
          "tp90" : -1.4666395874346977E-6,
          "mean" : -8.847320429245079E-23,
          "count" : 50000.0,
          "positive" : 45000,
          "stdDev" : 2.5398198161637395E-5,
          "tp75" : -5.381584365534481E-6,
          "zeros" : 0
        } ],
        "totalBatches" : 280,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 0.002428341505388512,
          "tp90" : "NaN",
          "count" : 7840.0,
          "positive" : 4337,
          "tp75" : "NaN",
          "zeros" : 734,
          "meanExponent" : -3.6138917401534116,
          "negative" : 2769,
          "min" : -0.0027476858343741697,
          "mean" : 9.421327177553196E-5,
          "stdDev" : 4.302979668534257E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.5115882213992125,
          "tp50" : -0.8035601632418563,
          "negative" : 4210,
          "min" : -3.9214813341847448,
          "max" : 15.95596665611611,
          "tp90" : 0.12821538848111008,
          "mean" : 4.284341332946548,
          "count" : 50000.0,
          "positive" : 45790,
          "stdDev" : 3.434333046999925,
          "tp75" : -0.1922416933957417,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "BiasLayer/e1035fb9-1fe3-4846-a360-62290000005b" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.017669923655000006,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 3.928290684428574E-6,
        "totalItems" : 1400000,
        "backp
```
...[skipping 744 bytes](etc/2.txt)...
```
       "meanExponent" : "NaN",
          "negative" : 0,
          "min" : -7.850123976632725E-27,
          "mean" : 3.476651661017923E-29,
          "stdDev" : 1.3674106656668975E-27
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 2.1115379266613696,
          "tp50" : -7.850123976632725E-27,
          "negative" : 0,
          "min" : 0.0,
          "max" : 0.0,
          "tp90" : -7.850123976632725E-27,
          "mean" : 32.963244897959186,
          "count" : 3920000.0,
          "positive" : 742538,
          "stdDev" : 78.1935837774458,
          "tp75" : -7.850123976632725E-27,
          "zeros" : 3177462
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/e1035fb9-1fe3-4846-a360-62290000005d" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.003204333660000001,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 1.6430897314285708E-7,
        "totalItems" : 1400000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -3.5392488423472597,
          "tp50" : -2.200839076264556E-4,
          "negative" : 5000,
          "min" : -2.8337685881496815,
          "max" : 0.0,
          "tp90" : -2.01477474198177E-4,
          "mean" : -1.5908411354585144E-4,
          "count" : 50000.0,
          "positive" : 0,
          "stdDev" : 0.013769651636084013,
          "tp75" : -2.0553039582404394E-4,
          "zeros" : 45000
        } ],
        "totalBatches" : 280,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.6804269112242998,
          "tp50" : 3.5883168461812023E-6,
          "negative" : 0,
          "min" : 2.294213405797219E-9,
          "max" : 0.9949922081057182,
          "tp90" : 1.4345205489626944E-5,
          "mean" : 0.1,
          "count" : 50000.0,
          "positive" : 50000,
          "stdDev" : 0.2538125621081272,
          "tp75" : 9.26048156237592E-6,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:201](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L201) executed in 0.78 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    89.82
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:208](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L208) executed in 0.30 seconds: 
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
![[5]](etc/test.2.png)  | 6 (63.7%), 4 (12.8%), 2 (11.6%)
![[4]](etc/test.3.png)  | 6 (51.0%), 0 (27.2%), 2 (7.9%) 
![[3]](etc/test.4.png)  | 2 (58.9%), 3 (27.1%), 8 (5.5%) 
![[9]](etc/test.5.png)  | 7 (47.8%), 9 (42.2%), 8 (5.3%) 
![[2]](etc/test.6.png)  | 7 (79.1%), 2 (12.0%), 9 (6.5%) 
![[7]](etc/test.7.png)  | 4 (54.0%), 9 (24.9%), 7 (16.7%)
![[2]](etc/test.8.png)  | 9 (41.4%), 4 (14.2%), 2 (13.2%)
![[2]](etc/test.9.png)  | 3 (68.0%), 2 (22.1%), 5 (4.4%) 
![[9]](etc/test.10.png) | 4 (34.1%), 3 (30.3%), 9 (16.8%)
![[3]](etc/test.11.png) | 8 (31.3%), 3 (17.4%), 2 (17.3%)




