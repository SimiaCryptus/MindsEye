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
    PipelineNetwork/3cc8990a-29bd-4377-9ee9-86380000002e
```



### Training
Code from [MomentumTest.java:42](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/orient/MomentumTest.java#L42) executed in 300.07 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 1000);
    return new IterativeTrainer(trainable)
      .setMonitor(monitor)
      .setOrientation(new ValidatingOrientationWrapper(new MomentumStrategy(new GradientDescent()).setCarryOver(0.8)))
      .setTimeout(5, TimeUnit.MINUTES)
      .setMaxIterations(500)
      .run();
```
Logging: 
```
    Constructing line search parameters: 
    Returning cached value; 2 buffers unchanged since 0.0 => 2.4557912005914484
    -450696.9589944947 vs (-447535.6685014313, -448394.07826924947); probe=0.001
    -450310.127492846 vs (-447535.6685014313, -447621.5164506608); probe=1.0E-4
    -450267.57345145376 vs (-447535.6685014313, -447536.52698846953); probe=1.0E-6
    th(0)=2.4557912005914484;dx=-447535.6685014313
    8.727634549517555E-76 vs (2.2586060319054417E-77, 0.0); probe=0.001
    8.727634549517553E-75 vs (2.2586060319054417E-77, 0.0); probe=1.0E-4
    8.727634549517556E-73 vs (2.2586060319054417E-77, 0.0); probe=1.0E-6
    Armijo: th(2.154434690031884)=23.596892033002977; dx=2.2586060319054417E-77 delta=-21.141100832411528
    4.6358725741243257E-32 vs (1.199707629788532E-33, 0.0); probe=0.001
    4.635872574124325E-31 vs (1.199707629788532E-33, 0.0); probe=1.0E-4
    4.635872574124326E-29 vs (1.199707629788532E-33, 0.0); probe=1.0E-6
    Armijo: th(1.077217345015942)=23.596892033002977; dx=1.199707629788532E-33 delta=-21.141100832411528
    0.0 vs (1.7548483386161032E-4, 0.0); probe=0.001
    0.0 vs (1.7548483386161032E-4, 0.0); probe=1.0E-4
    0.0 vs (1.7548483386161032E-4, 2.093896549156208E-15); probe=1.0E-6
    Armijo: th(0.3590724483386473)=23.596892033002977; dx=1.7548483386161032E-4 delta=-21.141100832411528
    0.14454682720320824 vs (0.18707555052934283, 8.790484715369376E-4); probe=0.001
    0.18708329799619755 vs (0.18707555052934283, 0.1870866516295901); probe=1.0E-4
    0.18707539846241975 vs (0.18707555052934283, 0.1870755071145986); probe=1.0E-6
    Armijo: th(0.08976811208466183)=23.57867360494217; dx=0.18707555052934283 delta=-21.122882404350722
    0.18659326646100868 vs (0.17686965982628083, 0.18706965776400047); probe=0.001
    0.18260788563029723 vs (0.17686965982628083, 0.18605763683686805); probe=1.0E-4
    0.17539205698267085 vs (0.17686965982628083, 0.17647958841446137); probe=1.0E-6
    Armijo: th(0.017953622416932366)=23.572020873000852; dx=0.17686965982628083 delta=-21.116229672409403
    45.62721117859417 vs (62.019902338078566, 38.27369728122325); probe=
```
...[skipping 178971 bytes](etc/1.txt)...
```
     (-1647.0968419062667, -1912.2565270963582); probe=0.001
    -1514.801179307076 vs (-1647.0968419062667, -1673.6674384735343); probe=1.0E-4
    -1501.7179957774085 vs (-1647.0968419062667, -1647.3626086114314); probe=1.0E-6
    New Minimum: 0.30112936020456726 > 0.28755833872758746
    END: th(1.0310976629360556E-5)=0.28755833872758746; dx=-1647.0968419062667 delta=0.02715212719555188
    Iteration 147 complete. Error: 0.28755833872758746 Total: 13333049299127.9570; Orientation: 0.0004; Line Search: 1.8341
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3093438235618535
    -5209.738781534829 vs (-5242.079357520921, -5316.875119388249); probe=0.001
    -5176.183233026067 vs (-5242.079357520921, -5249.5567510639285); probe=1.0E-4
    -5172.49294263303 vs (-5242.079357520921, -5242.154129054007); probe=1.0E-6
    th(0)=0.3093438235618535;dx=-5242.079357520921
    10330.783138189992 vs (10365.789736838708, 10423.802745271652); probe=0.001
    10305.149610653762 vs (10365.789736838708, 10371.586024628217); probe=1.0E-4
    10302.331669124425 vs (10365.789736838708, 10365.847694201293); probe=1.0E-6
    Armijo: th(2.221432573840241E-5)=0.33249845304793135; dx=10365.789736838708 delta=-0.023154629486077827
    1950.928864122861 vs (1801.5805165555485, 2024.1964618860973); probe=0.001
    1851.5007110957154 vs (1801.5805165555485, 1823.777767976074); probe=1.0E-4
    1840.5861693559732 vs (1801.5805165555485, 1801.8024194522754); probe=1.0E-6
    New Minimum: 0.3093438235618535 > 0.29995348544204725
    WOLF (strong): th(1.1107162869201204E-5)=0.29995348544204725; dx=1801.5805165555485 delta=0.009390338119806274
    -2918.711627127249 vs (-2916.242068127257, -3045.594598182561); probe=0.001
    -2860.717420564182 vs (-2916.242068127257, -2929.175907689576); probe=1.0E-4
    -2854.3386590258806 vs (-2916.242068127257, -2916.3714051587194); probe=1.0E-6
    END: th(3.702387623067068E-6)=0.3019197709189721; dx=-2916.242068127257 delta=0.007424052642881429
    Iteration 148 complete. Error: 0.29995348544204725 Total: 13335721763497.9510; Orientation: 0.0004; Line Search: 2.5013
    
```

Returns: 

```
    0.29995348544204725
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
      "BiasLayer/3cc8990a-29bd-4377-9ee9-86380000002f" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.014204729608719982,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 1.9368195828954704E-5,
        "totalItems" : 1789000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -7.8514848954475704,
          "tp50" : -4.125932924871553E-6,
          "negative" : 187746,
          "min" : -1.1103284661724431E-6,
          "max" : 1.1889794964216019E-6,
          "tp90" : -3.475002388100294E-6,
          "mean" : 6.236319255472279E-9,
          "count" : 392000.0,
          "positive" : 204254,
          "stdDev" : 4.3458172014308854E-7,
          "tp75" : -3.6427893213942644E-6,
          "zeros" : 0
        } ],
        "totalBatches" : 3578,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 6.248366055436451E-8,
          "tp90" : "NaN",
          "count" : 784.0,
          "positive" : 398,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -7.957107291564144,
          "negative" : 386,
          "min" : -8.315831307655746E-8,
          "mean" : -1.266603692968643E-9,
          "stdDev" : 2.120829960414308E-8
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -5.994825911600732,
          "tp50" : -8.315831307655746E-8,
          "negative" : 148419,
          "min" : 2.2005334282045204E-8,
          "max" : 2.2005334282045204E-8,
          "tp90" : -7.806477657752493E-8,
          "mean" : 34.062701529345816,
          "count" : 392000.0,
          "positive" : 243581,
          "stdDev" : 79.29065910698614,
          "tp75" : -7.806477657752493E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/3cc8990a-29bd-4377-9ee9-863800000031" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.002157133047512584,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 1.229717187255448E-6,
        "totalItems" : 1789000,
     
```
...[skipping 777 bytes](etc/2.txt)...
```
    42090208136,
          "tp90" : 2.7078315954966872E-8,
          "mean" : 0.1,
          "count" : 5000.0,
          "positive" : 5000,
          "stdDev" : 0.27393203524798526,
          "tp75" : 1.12558537319954E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "FullyConnectedLayer/3cc8990a-29bd-4377-9ee9-863800000030" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.008819911306875351,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 4.9946140309670254E-5,
        "totalItems" : 1789000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.783544685667208,
          "tp50" : -5.893675050598187E-5,
          "negative" : 500,
          "min" : -0.0019964091047420993,
          "max" : 0.0018598212510864548,
          "tp90" : -1.3165382824932493E-6,
          "mean" : -1.0748572433119078E-22,
          "count" : 5000.0,
          "positive" : 4498,
          "stdDev" : 2.342491972462543E-4,
          "tp75" : -1.1155975963759554E-5,
          "zeros" : 2
        } ],
        "totalBatches" : 3578,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 0.00338676719403089,
          "tp90" : "NaN",
          "count" : 7840.0,
          "positive" : 4233,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -3.554941126409369,
          "negative" : 3607,
          "min" : -0.0031995893144551217,
          "mean" : 5.229571869904092E-5,
          "stdDev" : 5.978927189870143E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.42393515059762193,
          "tp50" : -7.168247617916506,
          "negative" : 1893,
          "min" : -8.974214984014623,
          "max" : 14.560549903438247,
          "tp90" : -5.342709668481469,
          "mean" : 1.4280083998384097,
          "count" : 5000.0,
          "positive" : 3107,
          "stdDev" : 5.216287911365119,
          "tp75" : -5.901978210718633,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:211](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L211) executed in 0.60 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    91.97999999999999
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:218](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L218) executed in 0.07 seconds: 
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
![[5]](etc/test.2.png)  | 6 (99.6%), 4 (0.1%), 2 (0.1%)  
![[4]](etc/test.3.png)  | 6 (47.1%), 0 (34.9%), 5 (15.6%)
![[3]](etc/test.4.png)  | 2 (60.6%), 3 (34.6%), 8 (4.3%) 
![[6]](etc/test.5.png)  | 2 (58.6%), 7 (18.2%), 6 (10.9%)
![[7]](etc/test.6.png)  | 4 (58.7%), 9 (27.6%), 7 (12.9%)
![[2]](etc/test.7.png)  | 9 (86.7%), 4 (4.5%), 8 (3.4%)  
![[9]](etc/test.8.png)  | 3 (46.7%), 9 (24.4%), 4 (17.4%)
![[3]](etc/test.9.png)  | 8 (48.4%), 3 (29.6%), 5 (8.9%) 
![[5]](etc/test.10.png) | 7 (44.3%), 5 (37.0%), 0 (8.7%) 
![[6]](etc/test.11.png) | 5 (75.7%), 6 (17.0%), 8 (4.3%) 




