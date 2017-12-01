# ConvolutionLayer
## IrregularTest
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
      "id": "f4569375-56fe-4e46-925c-95f400000412",
      "isFrozen": false,
      "name": "ConvolutionLayer/f4569375-56fe-4e46-925c-95f400000412",
      "filter": {
        "dimensions": [
          3,
          3,
          18
        ],
        "data": [
          1.844,
          -0.12,
          -1.988,
          0.016,
          1.076,
          -1.28,
          1.292,
          -0.772,
          1.764,
          -0.544,
          1.904,
          1.028,
          -1.552,
          0.94,
          -1.396,
          -1.704,
          0.892,
          -1.004,
          1.68,
          -1.384,
          -0.932,
          -1.388,
          1.78,
          1.244,
          1.412,
          -0.32,
          -1.404,
          -0.572,
          1.444,
          -1.888,
          0.216,
          -0.164,
          1.916,
          -1.72,
          1.328,
          -1.964,
          -0.588,
          0.44,
          1.56,
          -1.604,
          -0.388,
          0.108,
          2.0,
          0.96,
          -1.108,
          -0.964,
          -0.676,
          -1.688,
          -0.964,
          -1.352,
          1.792,
          -0.164,
          1.148,
          1.116,
          -1.616,
          0.936,
          1.008,
          1.808,
          0.2,
          1.408,
          -0.916,
          1.98,
          1.836,
          -0.668,
          -0.644,
          -1.0,
          1.216,
          -1.108,
          1.108,
          0.364,
          -1.312,
          -1.512,
          -0.796,
          -1.392,
          0.332,
          -0.912,
          0.092,
          1.928,
          1.336,
          -1.492,
          0.512,
          -0.376,
          -1.36,
          0.228,
          0.74,
          0.824,
          -0.872,
          -1.7,
          1.628,
          -0.92,
          -1.548,
          -0.984,
          1.496,
          -1.492,
          1.544,
          -1.888,
          0.46,
          -1.372,
          -0.784,
          1.164,
          1.524,
          1.928,
          0.752,
          0.876,
          0.196,
          -0.504,
          1.356,
          -0.604,
          1.856,
          -1.308,
          0.756,
          0.072,
          1.268,
          0.056,
          1.216,
          -0.58,
          -0.636,
          -0.98,
          -1.94,
          -0.112,
          0.408,
          0.976,
          0.596,
          -1.936,
          1.02,
          0.564,
          1.316,
          -0.336,
          -1.388,
          0.176,
          -0.1,
          -0.072,
          -1.344,
          1.372,
          -0.016,
          0.704,
          -1.22,
          -0.652,
          -0.68,
          1.488,
          -1.248,
          -1.908,
          -0.676,
          0.48,
          -1.536,
          -1.776,
          -1.272,
          -0.34,
          0.436,
          -1.02,
          1.5,
          -0.768,
          1.06,
          -0.924,
          -1.804,
          -1.456,
          -1.832,
          0.312,
          -0.34,
          -1.544,
          1.18,
          1.516
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
    	[ [ 1.748, -0.724, 0.624 ], [ -0.14, 0.772, -0.268 ], [ 0.744, 0.412, -0.6 ] ],
    	[ [ 1.992, -0.668, -1.524 ], [ -1.148, -1.9, 1.824 ], [ 1.372, -1.408, -0.572 ] ],
    	[ [ 0.212, -1.212, 1.932 ], [ -1.236, -0.84, 1.752 ], [ 0.568, 0.756, 0.208 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 6.670448, -1.6089440000000004, 1.7518560000000003, 2.747568, -2.7350880000000006, -3.469151999999998 ], [ -0.6460160000000004, 4.495456, 10.112736, -0.708032, 12.007359999999998, 0.01691199999999958 ], [ 1.8244960000000001, -8.09536, -7.384240000000002, 1.9444959999999993, -1.241376000000001, -3.2662080000000002 ] ],
    	[ [ -6.0359039999999995, -8.629968, 10.798863999999998, -0.5356160000000011, -1.9996800000000003, -10.330912000000001 ], [ 1.6106240000000003, 5.587616000000001, -5.604512000000001, -0.008351999999999214, 9.525392000000002, -3.1807679999999996 ], [ 0.1646400000000002, 2.0342879999999997, 3.770288, -6.084192000000001, -6.011648, -6.858736 ] ],
    	[ [ -2.323728000000001, -5.697168, 1.571184, 5.9076319999999996, -7.862672, -4.775008000000001 ], [ -5.0191360000000005, 7.037312000000001, -9.259008, -10.728623999999996, -1.4688960000000002, -4.17688 ], [ -11.143247999999996, 2.923727999999999, 4.755472, 6.094591999999999, 7.742896000000001, 4.47736 ] ]
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
      "id": "f4569375-56fe-4e46-925c-95f40000041b",
      "isFrozen": false,
      "name": "ConvolutionLayer/f4569375-56fe-4e46-925c-95f40000041b",
      "filter": {
        "dimensions": [
          3,
          3,
          18
        ],
        "data": [
          1.844,
          -0.12,
          -1.988,
          0.016,
          1.076,
          -1.28,
          1.292,
          -0.772,
          1.764,
          -0.544,
          1.904,
          1.028,
          -1.552,
          0.94,
          -1.396,
          -1.704,
          0.892,
          -1.004,
          1.68,
          -1.384,
          -0.932,
          -1.388,
          1.78,
          1.244,
          1.412,
          -0.32,
          -1.404,
          -0.572,
          1.444,
          -1.888,
          0.216,
          -0.164,
          1.916,
          -1.72,
          1.328,
          -1.964,
          -0.588,
          0.44,
          1.56,
          -1.604,
          -0.388,
          0.108,
          2.0,
          0.96,
          -1.108,
          -0.964,
          -0.676,
          -1.688,
          -0.964,
          -1.352,
          1.792,
          -0.164,
          1.148,
          1.116,
          -1.616,
          0.936,
          1.008,
          1.808,
          0.2,
          1.408,
          -0.916,
          1.98,
          1.836,
          -0.668,
          -0.644,
          -1.0,
          1.216,
          -1.108,
          1.108,
          0.364,
          -1.312,
          -1.512,
          -0.796,
          -1.392,
          0.332,
          -0.912,
          0.092,
          1.928,
          1.336,
          -1.492,
          0.512,
          -0.376,
          -1.36,
          0.228,
          0.74,
          0.824,
          -0.872,
          -1.7,
          1.628,
          -0.92,
          -1.548,
          -0.984,
          1.496,
          -1.492,
          1.544,
          -1.888,
          0.46,
          -1.372,
          -0.784,
          1.164,
          1.524,
          1.928,
          0.752,
          0.876,
          0.196,
          -0.504,
          1.356,
          -0.604,
          1.856,
          -1.308,
          0.756,
          0.072,
          1.268,
          0.056,
          1.216,
          -0.58,
          -0.636,
          -0.98,
          -1.94,
          -0.112,
          0.408,
          0.976,
          0.596,
          -1.936,
          1.02,
          0.564,
          1.316,
          -0.336,
          -1.388,
          0.176,
          -0.1,
          -0.072,
          -1.344,
          1.372,
          -0.016,
          0.704,
          -1.22,
          -0.652,
          -0.68,
          1.488,
          -1.248,
          -1.908,
          -0.676,
          0.48,
          -1.536,
          -1.776,
          -1.272,
          -0.34,
          0.436,
          -1.02,
          1.5,
          -0.768,
          1.06,
          -0.924,
          -1.804,
          -1.456,
          -1.832,
          0.312,
          -0.34,
          -1.544,
          1.18,
          1.516
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
    	[ [ 1.748, -0.724, 0.624 ], [ -0.14, 0.772, -0.268 ], [ 0.744, 0.412, -0.6 ] ],
    	[ [ 1.992, -0.668, -1.524 ], [ -1.148, -1.9, 1.824 ], [ 1.372, -1.408, -0.572 ] ],
    	[ [ 0.212, -1.212, 1.932 ], [ -1.236, -0.84, 1.752 ], [ 0.568, 0.756, 0.208 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (54#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (54#)
    
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
    	[ [ 1.748, -0.724, 0.624 ], [ -0.14, 0.772, -0.268 ], [ 0.744, 0.412, -0.6 ] ],
    	[ [ 1.992, -0.668, -1.524 ], [ -1.148, -1.9, 1.824 ], [ 1.372, -1.408, -0.572 ] ],
    	[ [ 0.212, -1.212, 1.932 ], [ -1.236, -0.84, 1.752 ], [ 0.568, 0.756, 0.208 ] ]
    ]
    Output: [
    	[ [ 6.670448, -1.6089440000000004, 1.7518560000000003, 2.747568, -2.7350880000000006, -3.469151999999998 ], [ -0.6460160000000004, 4.495456, 10.112736, -0.708032, 12.007359999999998, 0.01691199999999958 ], [ 1.8244960000000001, -8.09536, -7.384240000000002, 1.9444959999999993, -1.241376000000001, -3.2662080000000002 ] ],
    	[ [ -6.0359039999999995, -8.629968, 10.798863999999998, -0.5356160000000011, -1.9996800000000003, -10.330912000000001 ], [ 1.6106240000000003, 5.587616000000001, -5.604512000000001, -0.008351999999999214, 9.525392000000002, -3.1807679999999996 ], [ 0.1646400000000002, 2.0342879999999997, 3.770288, -6.084192000000001, -6.011648, -6.858736 ] ],
    	[ [ -2.323728000000001, -5.697168, 1.571184, 5.9076319999999996, -7.862672, -4.775008000000001 ], [ -5.0191360000000005, 7.037312000000001, -9.259008, -10.728623999999996, -1.4688960000000002, -4.17688 ], [ -11.143247999999996, 2.923727999999999, 4.755472, 6.094591999999999, 7.742896000000001, 4.47736 ] ]
    ]
    Measured: [ [ 1.0760000000065162, -1.2800000000012801, 0.0, -0.7720000000066563, 1.7639999999996547, 0.0, 0.0, 0.0, 0.0, 0.9399999999981645 ], [ 0.01600000000046009, 1.0759999999976344, -1.2800000000012801, 1.2919999999994047, -0.7720000000066563, 1.7639999999996547, 0.0, 0.0, 0.0, -1.5520000000002199 ], [ 0.0, 0.01600000000046009, 1.0760000000065162, 0.0, 1.2919999999994047, -0.7720000000066563, 0.0, 0.0, 0.0, 0.0 ], [ -0.11999999999900979, -1.988000000006096, 0.0, 1.0760000000065162, -1.2800000000012801, 0.0, -0.7720000000022154, 1.7639999999996547, 0.0, 1.9039999999970192 ], [ 1.8440000000019552, -0.11999999999900979, -1.9879999999972142, 0.01600000000046009, 1.0759999999976344, -1.2800000000012801, 1.2919999999994047, -0.7720000000022154, 1.763999999990773, -0.5
```
...[skipping 5961 bytes](etc/1.txt)...
```
    14, -1.148, -1.236, 0.744, 1.372, 0.568, 0.0 ], [ 0.0, 1.748, 1.992, 0.0, -0.14, -1.148, 0.0, 0.744, 1.372, 0.0 ], [ 0.0, 0.0, 0.0, 1.992, 0.212, 0.0, -1.148, -1.236, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.748, 1.992, 0.212, -0.14, -1.148, -1.236, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 1.748, 1.992, 0.0, -0.14, -1.148, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.148 ] ]
    Error: [ [ 4.735989378445993E-12, -2.23532303778029E-12, 0.0, 1.7050805212193154E-12, -5.871192421125215E-12, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 7.076450536658285E-12, -4.1457948185552596E-12, -2.23532303778029E-12, -8.102407633714392E-13, 1.7050805212193154E-12, 3.010591775876037E-12, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -1.805333660342967E-12, 4.735989378445993E-12, 0.0, -8.102407633714392E-13, 1.7050805212193154E-12, 0.0, 0.0, 0.0, 0.0 ], [ -4.503064587879635E-13, -5.651312751098203E-13, 0.0, 4.735989378445993E-12, -2.23532303778029E-12, 0.0, 1.7050805212193154E-12, -1.4303003226245892E-12, 0.0, 0.0 ], [ -8.053557820630886E-13, -4.503064587879635E-13, -5.651312751098203E-13, 7.076450536658285E-12, -4.1457948185552596E-12, -2.23532303778029E-12, -8.102407633714392E-13, -2.7358115772813107E-12, -5.871192421125215E-12, 0.0 ], [ 0.0, -8.053557820630886E-13, -4.503064587879635E-13, 0.0, -1.805333660342967E-12, -4.1457948185552596E-12, 0.0, -8.102407633714392E-13, -7.176703675781937E-12, 0.0 ], [ 0.0, 0.0, 0.0, -4.503064587879635E-13, -5.651312751098203E-13, 0.0, 2.950972799453666E-13, -2.23532303778029E-12, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -8.053557820630886E-13, -4.503064587879635E-13, -5.651312751098203E-13, -1.805333660342967E-12, 2.950972799453666E-13, -2.23532303778029E-12, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -8.053557820630886E-13, -4.503064587879635E-13, 0.0, -1.805333660342967E-12, -4.1457948185552596E-12, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.950972799453666E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.7048e-13 +- 1.8006e-12 [0.0000e+00 - 1.7203e-11] (10206#)
    relativeTol: 3.0820e-12 +- 8.1306e-12 [6.2513e-17 - 2.6318e-10] (1764#)
    
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



