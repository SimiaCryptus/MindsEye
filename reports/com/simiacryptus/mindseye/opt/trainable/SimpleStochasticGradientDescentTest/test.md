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
    PipelineNetwork/3cc8990a-29bd-4377-9ee9-86380000008a
```



### Training
Training a model involves a few different components. First, our model is combined mapCoords a loss function. Then we take that model and combine it mapCoords our training data to define a trainable object. Finally, we use a simple iterative scheme to refine the weights of our model. The final output is the last output value of the loss function when evaluating the last batch.

Code from [SimpleStochasticGradientDescentTest.java:47](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/opt/trainable/SimpleStochasticGradientDescentTest.java#L47) executed in 300.55 seconds: 
```java
    SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
    Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 10000);
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
    Returning cached value; 2 buffers unchanged since 0.0 => 2.3429209183014015
    th(0)=2.3429209183014015;dx=-393994.18618268595
    Armijo: th(2.154434690031884)=20.65971449221498; dx=-3.617123722910942E-8 delta=-18.31679357391358
    Armijo: th(1.077217345015942)=20.65904923791293; dx=0.0039136076351916505 delta=-18.31612831961153
    Armijo: th(0.3590724483386473)=20.65766314649771; dx=0.006312553015012954 delta=-18.31474222819631
    Armijo: th(0.08976811208466183)=20.65060295883967; dx=0.29992781121805207 delta=-18.307682040538268
    Armijo: th(0.017953622416932366)=20.60538854495738; dx=5.563229831322753 delta=-18.262467626655976
    Armijo: th(0.002992270402822061)=20.35443377180109; dx=238.24819914860214 delta=-18.01151285349969
    Armijo: th(4.2746720040315154E-4)=18.46735255336322; dx=10220.54080952758 delta=-16.124431635061818
    Armijo: th(5.343340005039394E-5)=8.153006988954418; dx=267989.78839904466 delta=-5.810086070653017
    New Minimum: 2.3429209183014015 > 1.833559977694276
    WOLF (strong): th(5.9370444500437714E-6)=1.833559977694276; dx=11275.87058160502 delta=0.5093609406071256
    END: th(5.937044450043771E-7)=2.23306348952676; dx=-345615.31001240114 delta=0.10985742877464144
    Iteration 1 complete. Error: 1.833559977694276 Total: 15361658607881.4080; Orientation: 0.0004; Line Search: 9.1465
    Returning cached value; 2 buffers unchanged since 0.0 => 2.235226073136479
    th(0)=2.235226073136479;dx=-315179.56107137114
    New Minimum: 2.235226073136479 > 2.0521611842828587
    END: th(1.2790974519435567E-6)=2.0521611842828587; dx=-257056.7867058916 delta=0.18306488885362038
    Iteration 2 complete. Error: 2.0521611842828587 Total: 15363587675773.3420; Orientation: 0.0002; Line Search: 0.9588
    Returning cached value; 2 buffers unchanged since 0.0 => 2.0561131964272152
    th(0)=2.0561131964272152;dx=-225360.86441922645
    New Minimum: 2.0561131964272152 > 1.7750003246373989
    END: th(2.755731922398589E-6)=1.7750003246373989; dx=-182663.38066881397 delta=0.28111287178981637
    Iteration 3 complete. Error: 
```
...[skipping 50024 bytes](etc/1.txt)...
```
    => 0.36577474699338564
    th(0)=0.36577474699338564;dx=-441.88092942788205
    New Minimum: 0.36577474699338564 > 0.3634810482945673
    END: th(1.547883683527132E-5)=0.3634810482945673; dx=-159.56624387319584 delta=0.002293698698818347
    Iteration 102 complete. Error: 0.3634810482945673 Total: 15641536485024.3800; Orientation: 0.0003; Line Search: 0.9618
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3457259686862541
    th(0)=0.3457259686862541;dx=-642.2060681580288
    Armijo: th(3.334814303925187E-5)=0.3503444520321339; dx=1173.4691532788663 delta=-0.004618483345879776
    New Minimum: 0.3457259686862541 > 0.34425844034943265
    WOLF (strong): th(1.6674071519625935E-5)=0.34425844034943265; dx=283.7402208021925 delta=0.0014675283368214553
    END: th(5.558023839875312E-6)=0.3443772681111409; dx=-328.2448388962697 delta=0.0013487005751132153
    Iteration 103 complete. Error: 0.34425844034943265 Total: 15645366961917.4630; Orientation: 0.0002; Line Search: 2.8457
    Returning cached value; 2 buffers unchanged since 0.0 => 0.34138784497870556
    th(0)=0.34138784497870556;dx=-449.34072560345237
    New Minimum: 0.34138784497870556 > 0.3395293985146668
    END: th(1.1974399368651587E-5)=0.3395293985146668; dx=-169.80080216820878 delta=0.0018584464640387655
    Iteration 104 complete. Error: 0.3395293985146668 Total: 15647281228583.6330; Orientation: 0.0003; Line Search: 0.9488
    Returning cached value; 2 buffers unchanged since 0.0 => 0.3604443179948655
    th(0)=0.3604443179948655;dx=-932.3399388949196
    Armijo: th(2.5798061392118866E-5)=0.36782617069663115; dx=2098.420722268801 delta=-0.007381852701765668
    New Minimum: 0.3604443179948655 > 0.3592584782505104
    WOLF (strong): th(1.2899030696059433E-5)=0.3592584782505104; dx=570.9812643495379 delta=0.0011858397443550994
    New Minimum: 0.3592584782505104 > 0.35897280930766096
    END: th(4.299676898686477E-6)=0.35897280930766096; dx=-431.74986901376616 delta=0.0014715086872045258
    Iteration 105 complete. Error: 0.35897280930766096 Total: 15651138830905.7190; Orientation: 0.0003; Line Search: 2.8726
    
```

Returns: 

```
    0.35897280930766096
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
      "FullyConnectedLayer/3cc8990a-29bd-4377-9ee9-86380000008c" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.007893168295512829,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 5.772788931794874E-6,
        "totalItems" : 3120000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.705445153082484,
          "tp50" : -1.3696882583495512E-5,
          "negative" : 5000,
          "min" : -1.9993656162889314E-4,
          "max" : 1.926309582483272E-4,
          "tp90" : -9.611902830316841E-7,
          "mean" : -5.966712434557456E-23,
          "count" : 50000.0,
          "positive" : 45000,
          "stdDev" : 2.4798901168307757E-5,
          "tp75" : -3.8128529285268954E-6,
          "zeros" : 0
        } ],
        "totalBatches" : 624,
        "weights" : [ "java.util.HashMap", {
          "tp50" : "NaN",
          "buffers" : 1,
          "max" : 0.002395070346289692,
          "tp90" : "NaN",
          "count" : 7840.0,
          "positive" : 4283,
          "tp75" : "NaN",
          "zeros" : 0,
          "meanExponent" : -3.6462493580260515,
          "negative" : 3557,
          "min" : -0.0018670615478225548,
          "mean" : 4.984667828186105E-5,
          "stdDev" : 4.2075989427386035E-4
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : 0.24763901524952303,
          "tp50" : -3.836438170905612,
          "negative" : 17682,
          "min" : -6.229141842627336,
          "max" : 11.49140574676917,
          "tp90" : -2.931541717975771,
          "mean" : 1.3806975734331013,
          "count" : 50000.0,
          "positive" : 32318,
          "stdDev" : 3.4734879281589968,
          "tp75" : -3.240388482158354,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "BiasLayer/3cc8990a-29bd-4377-9ee9-86380000008b" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.01208613412371795,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 2.9409560740384604E-6,
        "totalItems" : 3120000,
        "backpr
```
...[skipping 794 bytes](etc/2.txt)...
```
    gative" : 393,
          "min" : -1.3097996753892592E-8,
          "mean" : -1.9522996189739856E-10,
          "stdDev" : 4.153214491379428E-9
        } ],
        "class" : "com.simiacryptus.mindseye.layers.java.BiasLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -6.599718856108306,
          "tp50" : -1.3097996753892592E-8,
          "negative" : 1551959,
          "min" : 1.4890018398839344E-9,
          "max" : 1.4890018398839344E-9,
          "tp90" : -1.2998717572042378E-8,
          "mean" : 33.340635714092,
          "count" : 3920000.0,
          "positive" : 2368041,
          "stdDev" : 78.63668075484847,
          "tp75" : -1.2998717572042378E-8,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ],
      "SoftmaxActivationLayer/3cc8990a-29bd-4377-9ee9-86380000008d" : [ "java.util.HashMap", {
        "avgMsPerItem" : 0.0019512308304487193,
        "medianMsPerItem" : "NaN",
        "avgMsPerItem_Backward" : 9.035885474358984E-8,
        "totalItems" : 3120000,
        "backpropStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -3.5460229814248407,
          "tp50" : -2.1470386837690365E-4,
          "negative" : 5000,
          "min" : -0.630533213606342,
          "max" : 0.0,
          "tp90" : -2.0096583202481818E-4,
          "mean" : -0.006448663210231676,
          "count" : 50000.0,
          "positive" : 0,
          "stdDev" : 1.387832734365472,
          "tp75" : -2.0388695486472192E-4,
          "zeros" : 45000
        } ],
        "totalBatches" : 624,
        "class" : "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
        "outputStatistics" : [ "java.util.HashMap", {
          "meanExponent" : -2.8984495326403943,
          "tp50" : 1.2740163440748949E-6,
          "negative" : 0,
          "min" : 2.587380104583707E-10,
          "max" : 0.9932372493533416,
          "tp90" : 5.650243296234884E-6,
          "mean" : 0.1,
          "count" : 50000.0,
          "positive" : 50000,
          "stdDev" : 0.2602583700203713,
          "tp75" : 3.5058345457953074E-6,
          "zeros" : 0
        } ],
        "medianMsPerItem_Backward" : "NaN"
      } ]
    } ]
```



### Validation
If we run our model against the entire validation dataset, we get this accuracy:

Code from [MnistTestBase.java:211](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L211) executed in 0.66 seconds: 
```java
    return MNIST.validationDataStream().mapToDouble(labeledObject ->
      predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
      .average().getAsDouble() * 100;
```

Returns: 

```
    90.59
```



Let's examine some incorrectly predicted results in more detail:

Code from [MnistTestBase.java:218](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/mnist/MnistTestBase.java#L218) executed in 0.22 seconds: 
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
![[5]](etc/test.2.png)  | 6 (84.2%), 2 (6.6%), 0 (3.0%)  
![[4]](etc/test.3.png)  | 6 (44.7%), 0 (37.8%), 2 (5.9%) 
![[1]](etc/test.4.png)  | 3 (54.4%), 1 (32.6%), 8 (3.1%) 
![[3]](etc/test.5.png)  | 2 (59.0%), 3 (31.8%), 8 (4.8%) 
![[2]](etc/test.6.png)  | 7 (78.3%), 2 (11.3%), 9 (7.3%) 
![[9]](etc/test.7.png)  | 4 (35.8%), 9 (29.8%), 8 (18.2%)
![[7]](etc/test.8.png)  | 1 (41.9%), 7 (33.8%), 9 (8.6%) 
![[7]](etc/test.9.png)  | 4 (69.2%), 9 (22.6%), 7 (5.3%) 
![[2]](etc/test.10.png) | 9 (41.5%), 8 (17.8%), 4 (11.8%)
![[9]](etc/test.11.png) | 3 (32.7%), 4 (31.4%), 9 (17.8%)




