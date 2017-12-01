# ConvolutionLayer
## DownsizeTest
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
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "f4569375-56fe-4e46-925c-95f400000014",
      "isFrozen": false,
      "name": "ConvolutionLayer/f4569375-56fe-4e46-925c-95f400000014",
      "filter": {
        "dimensions": [
          3,
          3,
          21
        ],
        "data": [
          0.6,
          -1.64,
          0.964,
          -1.54,
          -0.296,
          0.604,
          1.272,
          0.616,
          -1.128,
          0.296,
          -0.788,
          0.096,
          -1.196,
          -0.236,
          1.804,
          1.132,
          -1.312,
          -0.836,
          1.624,
          0.724,
          0.18,
          1.488,
          -1.668,
          1.908,
          -1.696,
          0.016,
          0.22,
          -1.236,
          -0.88,
          -0.86,
          -0.14,
          0.352,
          0.844,
          0.516,
          0.716,
          1.172,
          -0.66,
          -0.556,
          -0.252,
          -0.068,
          -1.12,
          -0.44,
          0.568,
          0.156,
          0.468,
          -0.736,
          -0.936,
          -1.252,
          -1.292,
          1.06,
          -0.772,
          0.904,
          -1.604,
          0.772,
          -0.256,
          0.332,
          1.912,
          0.284,
          -0.488,
          -1.856,
          1.192,
          1.308,
          1.516,
          0.156,
          -1.096,
          -1.02,
          0.424,
          -1.324,
          -1.312,
          0.3,
          1.084,
          -1.288,
          -0.76,
          0.4,
          1.432,
          -1.364,
          -1.76,
          -1.628,
          0.532,
          0.612,
          0.816,
          1.852,
          0.624,
          -0.352,
          0.956,
          -1.576,
          0.992,
          1.476,
          -1.608,
          1.808,
          1.704,
          -1.176,
          -1.056,
          -1.36,
          0.12,
          1.932,
          1.812,
          1.396,
          -0.264,
          -1.28,
          -0.028,
          1.064,
          -1.276,
          -1.612,
          -1.916,
          1.808,
          0.62,
          -1.968,
          -1.628,
          0.264,
          0.008,
          -0.744,
          -1.664,
          -0.06,
          -1.156,
          0.884,
          1.532,
          0.604,
          -0.704,
          -0.416,
          -0.476,
          1.628,
          0.072,
          0.976,
          0.988,
          -1.728,
          1.68,
          0.924,
          1.508,
          1.668,
          1.308,
          0.052,
          -0.464,
          -1.136,
          0.28,
          0.204,
          1.356,
          1.332,
          -0.952,
          1.356,
          0.964,
          0.076,
          1.672,
          -1.836,
          -1.596,
          -0.176,
          -1.768,
          1.2,
          -0.944,
          1.672,
          -0.78,
          0.104,
          -1.92,
          1.544,
          -1.364,
          1.148,
          0.664,
          1.216,
          1.168,
          1.384,
          1.364,
          -1.048,
          -1.348,
          0.732,
          -0.576,
          1.156,
          0.332,
          1.116,
          -0.092,
          -0.732,
          0.428,
          0.292,
          0.428,
          -1.516,
          -0.596,
          -1.3,
          -0.456,
          -1.02,
          0.78,
          0.788,
          -0.288,
          1.784,
          -0.708,
          1.22,
          1.46,
          1.632,
          1.456,
          0.448,
          0.116
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.01 seconds: 
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
    	[ [ -1.672, 0.356, 0.744, 1.496, -0.4, -1.208, 0.224 ], [ -0.996, 0.468, -0.22, -0.488, 1.388, 0.636, -1.512 ], [ -0.924, -1.864, 0.668, -1.984, 1.908, 1.424, 0.464 ] ],
    	[ [ 0.976, 1.876, -1.172, -0.016, 0.432, -1.044, -1.516 ], [ 1.704, -1.908, -0.544, -0.412, 0.392, 1.552, -0.784 ], [ -1.904, 0.716, 1.936, -0.768, -0.012, -0.34, 0.432 ] ],
    	[ [ -0.176, 1.82, -0.96, 0.992, 1.156, 0.924, 0.22 ], [ -0.188, -1.88, -0.804, -0.916, 1.044, 1.124, 1.268 ], [ -0.432, -0.628, 0.872, 0.488, 0.696, -0.348, 0.264 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.2397280000000004, 3.687152, -8.059328 ] ]
    ]
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
    	[ [ -1.672, 0.356, 0.744, 1.496, -0.4, -1.208, 0.224 ], [ -0.996, 0.468, -0.22, -0.488, 1.388, 0.636, -1.512 ], [ -0.924, -1.864, 0.668, -1.984, 1.908, 1.424, 0.464 ] ],
    	[ [ 0.976, 1.876, -1.172, -0.016, 0.432, -1.044, -1.516 ], [ 1.704, -1.908, -0.544, -0.412, 0.392, 1.552, -0.784 ], [ -1.904, 0.716, 1.936, -0.768, -0.012, -0.34, 0.432 ] ],
    	[ [ -0.176, 1.82, -0.96, 0.992, 1.156, 0.924, 0.22 ], [ -0.188, -1.88, -0.804, -0.916, 1.044, 1.124, 1.268 ], [ -0.432, -0.628, 0.872, 0.488, 0.696, -0.348, 0.264 ] ]
    ]
    Output: [
    	[ [ 1.2397280000000004, 3.687152, -8.059328 ] ]
    ]
    Measured: [ [ 0.5999999999994898, 0.2959999999996299, 1.6240000000067312 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ -1.2360000000000149, -0.6600000000034356, -0.735999999985637 ] ]
    Implemented: [ [ 0.6, 0.296, 1.624 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ -1.236, -0.66, -0.736 ] ]
    Error: [ [ -5.101474798152594E-13, -3.7009284525879593E-13, 6.731060153697399E-12 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ -1.4876988529977098E-14, -3.435585149702547E-12, 1.436295526957565E-11 ] ]
    Learning Gradient for weight set 0
    Inputs: [
    	[ [ -1.672, 0.356, 0.744, 1.496, -0.4, -1.208, 0.224 ], [ -0.996, 0.468, -0.22, -0.488, 1.388, 0.636, -1.512 ], [ -0.924, -1.864, 0.668, -1.984, 1.908, 1.424, 0.464 ] ],
    	[ [ 0.976, 1.876, -1.172, -0.016, 0.432, -1.044, -1.516 ], [ 1.704, -1.908, -0.544, -0.412, 0.392, 1.552, -0.784 ], [ -1.904, 0.716, 1.936, -0.768, -0.012, -0.34, 0.432 ] ],
    	[ [ -0.176, 1.82, -0.96, 0.992, 1.156, 0.924, 0.22 ], [ -0.188, -1.88, -0.804, -0.916, 1.044, 1.124, 1.268 ], [ -0.432, -0.628, 0.872, 0.488, 0.696, -0.348, 0.264 ] ]
    ]
    Outputs: [
    	[ [ 1.2397280000000004, 3.687152, -8.059328 ] ]
    ]
    Measured Gradient: [ [ -1.6719999999992297, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, -1.6719999999992297, 0.0 ] ]
    Implemented Gradient: [ [ -1.672, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, -1.672, 0.0 ] ]
    Error: [ [ 7.702727344849336E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 7.702727344849336E-13, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4492e-13 +- 9.8894e-13 [0.0000e+00 - 1.4363e-11] (756#)
    relativeTol: 2.4464e-12 +- 3.6201e-12 [6.0182e-15 - 1.6462e-11] (42#)
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:82)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:139)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:69)
    	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
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
    	at org.junit.runners.ParentRunner.run(ParentRunner.jav
```
...[skipping 879 bytes](etc/1.txt)...
```
    com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    Caused by: java.lang.RuntimeException: java.util.concurrent.ExecutionException: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:215)
    	at com.simiacryptus.util.lang.StaticResourcePool.apply(StaticResourcePool.java:88)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.run(GpuController.java:211)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.testUnFrozen(DerivativeTester.java:125)
    	at com.simiacryptus.mindseye.layers.DerivativeTester.test(DerivativeTester.java:92)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$15(LayerTestBase.java:140)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:157)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	... 36 more
    Caused by: java.util.concurrent.ExecutionException: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at java.util.concurrent.FutureTask.report(FutureTask.java:122)
    	at java.util.concurrent.FutureTask.get(FutureTask.java:192)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:213)
    	... 44 more
    Caused by: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$testUnFrozen$17(DerivativeTester.java:142)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$null$7(GpuController.java:213)
    	at java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:511)
    	at java.util.concurrent.FutureTask.run(FutureTask.java:266)
    	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    	at java.lang.Thread.run(Thread.java:748)
    
```



