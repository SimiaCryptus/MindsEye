# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer",
      "id": "f4569375-56fe-4e46-925c-95f40000016a",
      "isFrozen": false,
      "name": "ConvolutionLayer/f4569375-56fe-4e46-925c-95f40000016a",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          1.484,
          -0.112,
          -0.236,
          1.044
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ -0.396, -1.472 ], [ -1.464, 1.784 ], [ 1.856, -0.512 ] ],
    	[ [ -0.628, -1.416 ], [ -0.7, 1.668 ], [ -0.944, -0.556 ] ],
    	[ [ -0.024, 0.864 ], [ 1.604, -1.62 ], [ -1.524, 1.412 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.2402720000000001, -1.492416 ], [ -2.5936, 2.0264640000000003 ], [ 2.8751360000000004, -0.7424000000000001 ] ],
    	[ [ -0.5977760000000001, -1.4079679999999999 ], [ -1.432448, 1.819792 ], [ -1.26968, -0.47473600000000005 ] ],
    	[ [ -0.23951999999999998, 0.9047040000000001 ], [ 2.7626560000000002, -1.8709280000000001 ], [ -2.594848, 1.644816 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "f4569375-56fe-4e46-925c-95f400000171",
      "isFrozen": false,
      "name": "ConvolutionLayer/f4569375-56fe-4e46-925c-95f400000171",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          1.484,
          -0.112,
          -0.236,
          1.044
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ -0.396, -1.472 ], [ -1.464, 1.784 ], [ 1.856, -0.512 ] ],
    	[ [ -0.628, -1.416 ], [ -0.7, 1.668 ], [ -0.944, -0.556 ] ],
    	[ [ -0.024, 0.864 ], [ 1.604, -1.62 ], [ -1.524, 1.412 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (18#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (18#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ -0.396, -1.472 ], [ -1.464, 1.784 ], [ 1.856, -0.512 ] ],
    	[ [ -0.628, -1.416 ], [ -0.7, 1.668 ], [ -0.944, -0.556 ] ],
    	[ [ -0.024, 0.864 ], [ 1.604, -1.62 ], [ -1.524, 1.412 ] ]
    ]
    Output: [
    	[ [ -0.2402720000000001, -1.492416 ], [ -2.5936, 2.0264640000000003 ], [ 2.8751360000000004, -0.7424000000000001 ] ],
    	[ [ -0.5977760000000001, -1.4079679999999999 ], [ -1.432448, 1.819792 ], [ -1.26968, -0.47473600000000005 ] ],
    	[ [ -0.23951999999999998, 0.9047040000000001 ], [ 2.7626560000000002, -1.8709280000000001 ], [ -2.594848, 1.644816 ] ]
    ]
    Measured: [ [ 1.484000000000485, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.11200000000100019 ], [ 0.0, 1.4839999999993747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.4839999999999298, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.484000000000485, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.484000000000485, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.484000000000485, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.483999999996044, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.484000000000485, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.484000000000485, 0.0 ], [ -0.236000000000125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0439999999989347 ] ]
    Implemented: [ [ 1.484, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.112 ], [ 0.0, 1.484, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.484, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.484, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.484, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.484, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.484, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.484, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.484, 0.0 ], [ -0.236, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.044 ] ]
    Error: [ [ 4.849454171562684E-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0001860450969957E-12 ], [ 0.0, -6.252776074688882E-13, 0.0, 0.0, 0.0, 0.0, 
```
...[skipping 808 bytes](etc/1.txt)...
```
    0.864 ], [ 1.604, -1.62 ], [ -1.524, 1.412 ] ]
    ]
    Outputs: [
    	[ [ -0.2402720000000001, -1.492416 ], [ -2.5936, 2.0264640000000003 ], [ 2.8751360000000004, -0.7424000000000001 ] ],
    	[ [ -0.5977760000000001, -1.4079679999999999 ], [ -1.432448, 1.819792 ], [ -1.26968, -0.47473600000000005 ] ],
    	[ [ -0.23951999999999998, 0.9047040000000001 ], [ 2.7626560000000002, -1.8709280000000001 ], [ -2.594848, 1.644816 ] ]
    ]
    Measured Gradient: [ [ -0.3959999999991748, -0.628000000000295, -0.024000000000135024, -1.4640000000021303, -0.700000000000145, 1.6039999999994947, 1.855999999995639, -0.9440000000005, -1.5239999999971943, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.396000000000285 ], [ -1.4719999999998623, -1.4159999999996398, 0.8639999999998649, 1.7840000000024503, 1.6679999999991146, -1.6199999999999548, -0.5120000000014002, -0.556000000000445, 1.4119999999984145, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4720000000001399 ] ]
    Implemented Gradient: [ [ -0.396, -0.628, -0.024, -1.464, -0.7, 1.604, 1.856, -0.944, -1.524, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.396 ], [ -1.472, -1.416, 0.864, 1.784, 1.668, -1.62, -0.512, -0.556, 1.412, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.472 ] ]
    Error: [ [ 8.252287742038789E-13, -2.949862576429041E-13, -1.3502393647613076E-13, -2.1302959396507504E-12, -1.4499512701604544E-13, -5.053735208093713E-13, -4.36117808533254E-12, -5.000444502911705E-13, 2.8057556278326956E-12, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.849942504212777E-13 ], [ 1.376676550535194E-13, 3.601563491884008E-13, -1.3511414209688155E-13, 2.4502622153477205E-12, -8.852918398360998E-13, 4.529709940470639E-14, -1.4002132786572474E-12, -4.4497738826976274E-13, -1.5853984791647235E-12, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.3988810110276972E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5513e-13 +- 5.1355e-13 [0.0000e+00 - 4.3612e-12] (396#)
    relativeTol: 1.1446e-12 +- 2.5053e-12 [1.3981e-14 - 1.4378e-11] (72#)
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.AssertionError
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:82)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:139)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:69)
    	at sun.reflect.GeneratedMethodAccessor1.invoke(Unknown Source)
    	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    	at java.lang.reflect.Method.invoke(Method.java:498)
    	at org.junit.runners.model.FrameworkMethod$1.runReflectiveCall(FrameworkMethod.java:50)
    	at org.junit.internal.runners.model.ReflectiveCallable.run(ReflectiveCallable.java:12)
    	at org.junit.runners.model.FrameworkMethod.invokeExplosively(FrameworkMethod.java:47)
    	at org.junit.internal.runners.statements.InvokeMethod.evaluate(InvokeMethod.java:17)
    	at org.junit.runners.ParentRunner.runLeaf(ParentRunner.java:325)
    	at org.junit.runners.BlockJUnit4ClassRunner.runChild(BlockJUnit4ClassRunner.java:78)
    	at org.junit.runners.BlockJUnit4ClassRunner.runChild(BlockJUnit4ClassRunner.java:57)
    	at org.junit.runners.ParentRunner$3.run(ParentRunner.java:290)
    	at org.junit.runners.ParentRunner$1.schedule(ParentRunner.java:71)
    	at org.junit.runners.ParentRunner.runChildren(ParentRunner.java:288)
    	at org.junit.runners.ParentRunner.access$000(ParentRunner.java:58)
    	at org.junit.runners.ParentRunner$2.evaluate(ParentRunner.java:268)
    	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
    	at org.junit.runners.Suite.runChild(Suite.java:128)
    	at org.junit.runners.Suite.runChild(Suite.java:27)
    	at org.junit.runners
```
...[skipping 971 bytes](etc/2.txt)...
```
    troller.lambda$run$8(GpuController.java:215)
    	at com.simiacryptus.util.lang.StaticResourcePool.apply(StaticResourcePool.java:88)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.run(GpuController.java:211)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testUnFrozen(DerivativeTester.java:125)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:92)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$15(LayerTestBase.java:140)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:157)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	... 35 more
    Caused by: java.util.concurrent.ExecutionException: java.lang.AssertionError
    	at java.util.concurrent.FutureTask.report(FutureTask.java:122)
    	at java.util.concurrent.FutureTask.get(FutureTask.java:192)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:213)
    	... 43 more
    Caused by: java.lang.AssertionError
    	at com.simiacryptus.mindseye.layers.cudnn.CudaPtr.toDeviceAsDouble(CudaPtr.java:149)
    	at com.simiacryptus.mindseye.layers.cudnn.f64.ImgConcatLayer$1.accumulate(ImgConcatLayer.java:118)
    	at com.simiacryptus.mindseye.network.CountingNNResult.accumulate(CountingNNResult.java:110)
    	at com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer$1.accumulate(ConvolutionLayer.java:174)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$testUnFrozen$17(DerivativeTester.java:138)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$null$7(GpuController.java:213)
    	at java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:511)
    	at java.util.concurrent.FutureTask.run(FutureTask.java:266)
    	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    	at java.lang.Thread.run(Thread.java:748)
    
```



