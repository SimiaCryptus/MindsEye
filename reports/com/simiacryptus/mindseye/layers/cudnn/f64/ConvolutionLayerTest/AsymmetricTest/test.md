# ConvolutionLayer
## AsymmetricTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "id": "f4569375-56fe-4e46-925c-95f400000244",
      "isFrozen": false,
      "name": "ConvolutionLayer/f4569375-56fe-4e46-925c-95f400000244",
      "filter": {
        "dimensions": [
          1,
          1,
          8
        ],
        "data": [
          -1.008,
          -0.296,
          -1.064,
          1.34,
          -1.232,
          0.54,
          0.7,
          0.576
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    	[ [ -0.936, 0.4 ], [ 0.208, -1.808 ], [ -1.748, -0.592 ] ],
    	[ [ -0.276, -1.208 ], [ -1.764, -1.944 ], [ 0.016, 0.4 ] ],
    	[ [ 1.576, 1.392 ], [ 1.192, 1.908 ], [ 1.436, 0.252 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.4506880000000001, 0.49305600000000005, 1.2759040000000001, -1.0238400000000003 ], [ 2.017792, -1.0378880000000001, -1.486912, -0.7626879999999999 ], [ 2.4913279999999998, 0.197728, 1.4454720000000003, -2.683312 ] ],
    	[ [ 1.766464, -0.570624, -0.5519359999999999, -1.065648 ], [ 4.17312, -0.5276160000000001, 0.5160960000000002, -3.483504 ], [ -0.508928, 0.21126400000000004, 0.262976, 0.25184 ] ],
    	[ [ -3.303552, 0.285184, -0.7024640000000003, 2.913632 ], [ -3.552192, 0.6774880000000001, 0.06731199999999977, 2.696288 ], [ -1.757952, -0.28897599999999996, -1.351504, 2.069392 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "f4569375-56fe-4e46-925c-95f40000024d",
      "isFrozen": false,
      "name": "ConvolutionLayer/f4569375-56fe-4e46-925c-95f40000024d",
      "filter": {
        "dimensions": [
          1,
          1,
          8
        ],
        "data": [
          -1.008,
          -0.296,
          -1.064,
          1.34,
          -1.232,
          0.54,
          0.7,
          0.576
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
    	[ [ -0.936, 0.4 ], [ 0.208, -1.808 ], [ -1.748, -0.592 ] ],
    	[ [ -0.276, -1.208 ], [ -1.764, -1.944 ], [ 0.016, 0.4 ] ],
    	[ [ 1.576, 1.392 ], [ 1.192, 1.908 ], [ 1.436, 0.252 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (36#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (36#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ -0.936, 0.4 ], [ 0.208, -1.808 ], [ -1.748, -0.592 ] ],
    	[ [ -0.276, -1.208 ], [ -1.764, -1.944 ], [ 0.016, 0.4 ] ],
    	[ [ 1.576, 1.392 ], [ 1.192, 1.908 ], [ 1.436, 0.252 ] ]
    ]
    Output: [
    	[ [ 0.4506880000000001, 0.49305600000000005, 1.2759040000000001, -1.0238400000000003 ], [ 2.017792, -1.0378880000000001, -1.486912, -0.7626879999999999 ], [ 2.4913279999999998, 0.197728, 1.4454720000000003, -2.683312 ] ],
    	[ [ 1.766464, -0.570624, -0.5519359999999999, -1.065648 ], [ 4.17312, -0.5276160000000001, 0.5160960000000002, -3.483504 ], [ -0.508928, 0.21126400000000004, 0.262976, 0.25184 ] ],
    	[ [ -3.303552, 0.285184, -0.7024640000000003, 2.913632 ], [ -3.552192, 0.6774880000000001, 0.06731199999999977, 2.696288 ], [ -1.757952, -0.28897599999999996, -1.351504, 2.069392 ] ]
    ]
    Measured: [ [ -1.00800000000012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.296000000000185 ], [ 0.0, -1.00800000000012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.0080000000023404, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.0079999999978995, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.0080000000023404, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.0080000000023404, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0079999999978995, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.00800000000012, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.00800000000012, 0.0 ], [ -1.2319999999998998, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5399999999999849 ] ]
    Implemented: [ [ -1.008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.296 ], [ 0.0, -1.008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.008, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.008, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.008, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.008, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0,
```
...[skipping 1924 bytes](etc/1.txt)...
```
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.93600000000027 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.3999999999998449, -1.2080000000014302, 1.3920000000000599, -1.8079999999986995, -1.9439999999981694, 1.9079999999993547, -0.5919999999992598, 0.40000000000040004, 0.2520000000005851, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3999999999998449 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Implemented Gradient: [ [ -0.936, -0.276, 1.576, 0.208, -1.764, 1.192, -1.748, 0.016, 1.436, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.936 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.4, -1.208, 1.392, -1.808, -1.944, 1.908, -0.592, 0.4, 0.252, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Error: [ [ -2.6989521728637555E-13, -1.2752021660844548E-12, -3.530953307517848E-12, 1.5402956687893266E-12, 3.452793606584237E-13, -1.2503331703328513E-12, 8.053557820630886E-13, 4.600902991924727E-13, -8.952838470577262E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.6989521728637555E-13 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ -1.5509815654013437E-13, -1.4301893003221267E-12, 5.995204332975845E-14, 1.3005152510459084E-12, 1.830535723001958E-12, -6.45261621912141E-13, 7.401856905175919E-13, 4.000133557724439E-13, 5.850875339774575E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5509815654013437E-13 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.1267e-13 +- 4.2211e-13 [0.0000e+00 - 3.8558e-12] (936#)
    relativeTol: 6.8654e-13 +- 1.4825e-12 [1.3981e-14 - 1.4378e-11] (144#)
    
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



