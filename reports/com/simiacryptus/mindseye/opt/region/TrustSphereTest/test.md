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
    PipelineNetwork/3cc8990a-29bd-4377-9ee9-86380000005b
```



### Training
Code from [TrustSphereTest.java:43](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/region/TrustSphereTest.java#L43) executed in 180.83 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 10000);
    TrustRegionStrategy trustRegionStrategy = new TrustRegionStrategy() {
      @Override
      public TrustRegion getRegionPolicy(NNLayer layer) {
        return new AdaptiveTrustSphere();
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
    Returning cached value; 2 buffers unchanged since 0.0 => 2.522385411608557
    th(0)=2.522385411608557;dx=-556337.9404114333
    Armijo: th(2.154434690031884)=15.000881364147224; dx=1.3331208112907835E-4 delta=-12.478495952538667
    Armijo: th(1.077217345015942)=14.999622481396546; dx=0.0029084851718529695 delta=-12.47723706978799
    Armijo: th(0.3590724483386473)=14.995930029186239; dx=0.018683774747188527 delta=-12.473544617577682
    Armijo: th(0.08976811208466183)=14.986172669033191; dx=0.30788270666940765 delta=-12.463787257424634
    Armijo: th(0.017953622416932366)=14.92878097292419; dx=9.445983817730717 delta=-12.406395561315634
    Armijo: th(0.002992270402822061)=14.594662369353275; dx=283.2189327108041 delta=-12.072276957744718
    Armijo: th(4.2746720040315154E-4)=12.331053834108442; dx=12809.675076150881 delta=-9.808668422499885
    Armijo: th(5.343340005039394E-5)=4.613617265987955; dx=82219.28869392864 delta=-2.091231854379398
    New Minimum: 2.522385411608557 > 1.9523973525222282
    END: th(5.9370444500437714E-6)=1.9523973525222282; dx=-34894.94505857421 delta=0.5699880590863287
    Iteration 1 complete. Error: 1.9523973525222282 Total: 14329116362790.1450; Orientation: 0.0010; Line Search: 8.2615
    LBFGS Accumulation History: 2 points
    Returning cached value; 2 buffers unchanged since 0.0 => 1.9523973525222282
    th(0)=1.9523973525222282;dx=-285505.89795017126
    New Minimum: 1.9523973525222282 > 1.769105731976258
    WOLF (strong): th(1.279097451943557E-5)=1.769105731976258; dx=222041.49852365162 delta=0.18329162054597026
    New Minimum: 1.769105731976258 > 1.4586242579824629
    END: th(6.395487259717785E-6)=1.4586242579824629; dx=-25572.31443033462 delta=0.49377309453976537
    Iteration 2 complete. Error: 1.4586242579824629 Total: 14331096157776.8830; Orientation: 0.0006; Line Search: 1.9791
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 0.0 => 1.4586242579824629
    th(0)=1.4586242579824629;dx=-373241.8589243504
    Arm
```
...[skipping 50902 bytes](etc/1.txt)...
```
    anished. Popping history element from 0.3275544581500278, 0.32681214626768873, 0.32598289929329216, 0.32489164839288787
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.32489164839288787
    th(0)=0.32489164839288787;dx=-120.54357229773845
    New Minimum: 0.32489164839288787 > 0.3242930508434344
    WOLFE (weak): th(9.983424771805844E-6)=0.3242930508434344; dx=-119.39217135093607 delta=5.985975494534612E-4
    New Minimum: 0.3242930508434344 > 0.3237001884333859
    WOLFE (weak): th(1.996684954361169E-5)=0.3237001884333859; dx=-118.24677911294597 delta=0.0011914599595019837
    New Minimum: 0.3237001884333859 > 0.32138549559880836
    WOLFE (weak): th(5.990054863083506E-5)=0.32138549559880836; dx=-113.72448988368808 delta=0.003506152794079509
    New Minimum: 0.32138549559880836 > 0.31204981329633547
    END: th(2.3960219452334025E-4)=0.31204981329633547; dx=-94.48911996863313 delta=0.0128418350965524
    Iteration 72 complete. Error: 0.31204981329633547 Total: 14494844539790.0500; Orientation: 0.0009; Line Search: 3.8935
    Orientation vanished. Popping history element from 0.32681214626768873, 0.32598289929329216, 0.32489164839288787, 0.31204981329633547
    LBFGS Accumulation History: 3 points
    Returning cached value; 2 buffers unchanged since 0.0 => 0.31204981329633547
    th(0)=0.31204981329633547;dx=-493.48222505175147
    Armijo: th(5.162072796888517E-4)=2.450735343330173; dx=10625.949255587639 delta=-2.1386855300338374
    Armijo: th(2.5810363984442585E-4)=1.1469038169219317; dx=9234.85178459601 delta=-0.8348540036255963
    Armijo: th(8.603454661480862E-5)=0.43684349813801737; dx=5985.848670762155 delta=-0.1247936848416819
    Armijo: th(2.1508636653702154E-5)=0.3162921159601775; dx=1269.2769671154224 delta=-0.0042423026638420125
    New Minimum: 0.31204981329633547 > 0.31136956099231206
    END: th(4.301727330740431E-6)=0.31136956099231206; dx=-130.69398502115072 delta=6.802523040234099E-4
    Iteration 73 complete. Error: 0.31136956099231206 Total: 14499679026334.1560; Orientation: 0.0009; Line Search: 4.8335
    
```

Returns: 

```
    0.31136956099231206
```



Code from [MnistTestBase.java:141](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L141) executed in 0.02 seconds: 
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
Code from [MnistTestBase.java:154](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L154) executed in 0.55 seconds: 
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
      "BiasLayer/3cc8990a-29bd-4377-9ee9-86380000005c" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.0123644629718919,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 2.9884457872432444E-6,
        "totalItems" : 1850000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -8.61919879687295,
          "tp50" : -2.61407825096549E-7,
          "negative" : 1923152,
          "min" : -1.7238996561430895E-7,
          "max" : 1.403723921425994E-7,
          "tp90" : -2.2908514331880775E-7,
          "mean" : -3.685147330454043E-11,
          "count" : 3920000.0,
          "positive" : 1996848,
          "stdDev" : 3.230886431950583E-8,
          "tp75" : -2.3888612512356E-7,
          "zeros" : 0
        } ],
        "totalBatches" : 370,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 1.6132071328587044E-8,
          "tp90" : "NaN",
          "count" : 784.0,
          "positive" : 393,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -8.517115441479516,
          "negative" : 391,
          "min" : -2.3929892555283293E-8,
          "mean" : -8.223845032026919E-11,
          "stdDev" : 5.71228480152929E-9
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.50526963112392,
          "tp50" : -2.3929892555283293E-8,
          "negative" : 1539665,
          "min" : -8.16946532134799E-10,
          "max" : -8.16946532134799E-10,
          "tp90" : -2.2845796740621336E-8,
          "mean" : 33.127223724409845,
          "count" : 3920000.0,
          "positive" : 2380335,
          "stdDev" : 78.36686608206868,
          "tp75" : -2.2845796740621336E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "FullyConnectedLayer/3cc8990a-29bd-4377-9ee9-86380000005d" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.007888121055135129,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 5.920738684972974E-6,
        "totalItems" : 1850000,
      
```
...[skipping 791 bytes](etc/2.txt)...
```
         "negative" : 3524,
          "min" : -0.0022734729999018266,
          "mean" : 5.591399063086721E-5,
          "stdDev" : 4.553543338007737E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.2874659729403535,
          "tp50" : -4.395084018869146,
          "negative" : 17973,
          "min" : -7.845370915769476,
          "max" : 11.839400483517137,
          "tp90" : -3.389539491898236,
          "mean" : 1.4356291936417493,
          "count" : 50000.0,
          "positive" : 32027,
          "stdDev" : 3.813696026359685,
          "tp75" : -3.7257026070440125,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/3cc8990a-29bd-4377-9ee9-86380000005e" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.0019759984075675674,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 9.100811027027025E-8,
        "totalItems" : 1850000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -3.567065613111969,
          "tp50" : -2.1172027979151343E-4,
          "negative" : 5000,
          "min" : -0.1878489408282409,
          "max" : 0.0,
          "tp90" : -2.0064216499130738E-4,
          "mean" : -1.466384120639008E-4,
          "count" : 50000.0,
          "positive" : 0,
          "stdDev" : 0.006723525564216785,
          "tp75" : -2.0296527116215322E-4,
          "zeros" : 45000
        } ],
        "totalBatches" : 370,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -3.1292142219221435,
          "tp50" : 3.5276651009838435E-7,
          "negative" : 0,
          "min" : 1.147676057909109E-10,
          "max" : 0.9977162891411631,
          "tp90" : 1.8520069764107723E-6,
          "mean" : 0.1,
          "count" : 50000.0,
          "positive" : 50000,
          "stdDev" : 0.26490608089874695,
          "tp75" : 1.076410356070988E-6,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:211](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L211) executed in 0.65 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    90.97
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:218](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L218) executed in 0.05 seconds: 
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
![[5]](etc/test.2.png)  | 6 (92.6%), 4 (2.2%), 2 (2.1%)  
![[4]](etc/test.3.png)  | 6 (68.2%), 0 (17.4%), 5 (5.5%) 
![[3]](etc/test.4.png)  | 2 (57.1%), 3 (34.4%), 8 (4.6%) 
![[2]](etc/test.5.png)  | 7 (78.8%), 2 (14.4%), 9 (5.2%) 
![[3]](etc/test.6.png)  | 5 (48.5%), 3 (48.2%), 6 (1.9%) 
![[7]](etc/test.7.png)  | 1 (45.4%), 7 (36.8%), 9 (9.0%) 
![[7]](etc/test.8.png)  | 4 (58.4%), 9 (27.9%), 7 (10.3%)
![[2]](etc/test.9.png)  | 9 (56.3%), 8 (17.9%), 4 (8.2%) 
![[9]](etc/test.10.png) | 3 (34.5%), 4 (31.4%), 9 (18.0%)
![[3]](etc/test.11.png) | 8 (44.2%), 5 (16.1%), 3 (15.8%)




