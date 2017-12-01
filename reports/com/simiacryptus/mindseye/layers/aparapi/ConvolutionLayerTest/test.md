# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.04 seconds: 
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
      "id": "f4569375-56fe-4e46-925c-95f400000001",
      "isFrozen": false,
      "name": "ConvolutionLayer/f4569375-56fe-4e46-925c-95f400000001",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          -0.856,
          -0.692,
          -0.044,
          1.352,
          1.284,
          0.036,
          0.252,
          1.44,
          -0.128,
          1.3,
          0.264,
          1.644,
          -0.06,
          -0.46,
          0.636,
          -1.704,
          0.584,
          1.268,
          1.892,
          -0.836,
          -0.252,
          0.712,
          -1.94,
          0.116,
          -0.632,
          1.608,
          -0.416,
          -0.26,
          -0.96,
          1.384,
          -0.264,
          1.632,
          -1.74,
          -1.456,
          -0.752,
          0.872
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
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.29 seconds: 
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
    	[ [ -1.68, -0.12 ], [ -1.724, 0.88 ], [ 1.74, -1.104 ] ],
    	[ [ 0.792, -0.324 ], [ 0.644, 0.556 ], [ 0.712, -1.728 ] ],
    	[ [ 1.364, -0.012 ], [ 1.188, 0.368 ], [ 0.796, -1.368 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.12620799999999985, 0.007680000000000137 ], [ -9.022048, 3.1691520000000004 ], [ 2.851584000000001, -5.764144 ] ],
    	[ [ 2.0296319999999994, -2.3628 ], [ 0.7761439999999998, -1.4504800000000004 ], [ 6.046079999999999, -3.8274719999999993 ] ],
    	[ [ 0.4673920000000001, 2.20904 ], [ 3.874368, 1.3259679999999996 ], [ 5.489904, 2.579296 ] ]
    ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ -1.68, -0.12 ], [ -1.724, 0.88 ], [ 1.74, -1.104 ] ],
    	[ [ 0.792, -0.324 ], [ 0.644, 0.556 ], [ 0.712, -1.728 ] ],
    	[ [ 1.364, -0.012 ], [ 1.188, 0.368 ], [ 0.796, -1.368 ] ]
    ]
    Output: [
    	[ [ -0.12620799999999985, 0.007680000000000137 ], [ -9.022048, 3.1691520000000004 ], [ 2.851584000000001, -5.764144 ] ],
    	[ [ 2.0296319999999994, -2.3628 ], [ 0.7761439999999998, -1.4504800000000004 ], [ 6.046079999999999, -3.8274719999999993 ] ],
    	[ [ 0.4673920000000001, 2.20904 ], [ 3.874368, 1.3259679999999996 ], [ 5.489904, 2.579296 ] ]
    ]
    Measured: [ [ 1.2839999999991747, 0.03600000000325565, 0.0, 1.440000000005881, -0.12799999999923983, 0.0, 0.0, 0.0, 0.0, -0.45999999999990493 ], [ 1.3519999999989096, 1.2840000000036156, 0.03599999999992498, 0.2519999999961442, 1.4400000000014401, -0.12800000000368073, 0.0, 0.0, 0.0, -0.06000000000172534 ], [ 0.0, 1.3520000000033505, 1.284000000000285, 0.0, 0.2520000000005851, 1.4399999999969992, 0.0, 0.0, 0.0, 0.0 ], [ -0.6919999999999149, -0.0439999999990448, 0.0, 1.2840000000124974, 0.03599999999881476, 0.0, 1.4400000000014401, -0.12799999999479894, 0.0, 0.2639999999987097 ], [ -0.8560000000001899, -0.6919999999999149, -0.044000000000155026, 1.3520000000077914, 1.2839999999991747, 0.03600000000325565, 0.2519999999961442, 1.4399999999969992, -0.12800000000368073, 1.2999999999996348 ], [ 0.0, -0.8559999999979695, -0.6919999999999149, 0.0, 1.3519999999989096, 1.2840000000036156, 0.0, 0.252000000005026, 1.440000000005881, 0.0 ], [ 0.0, 0.0, 0.0, -0.6919999999865922, -0.0439999999990448, 0.0, 1.2839999999991747, 0.03600000000325565, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.8559999999846468, -0.6919999999999149, -0.0439999999990448, 1.3519999999989096, 1.2840000000036156, 0.03600000000325565, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.8559999999979695, -0.6920000000043558, 0.0, 1.3520000000077914, 1.2840000000036156, 0.0 ], [ -1.9399999999999973, 0.11600000000111521, 0.0, 1.6079999999973893, -0.4159999999997499, 0.0, 0.0, 0.0, 0.0, 1.6320000000000483 ] ]
    Implemented: [ [ 1.
```
...[skipping 4482 bytes](etc/1.txt)...
```
    .68, 0.792, 1.364, -1.724, 0.644, 1.188, 1.74, 0.712, 0.796, 0.0 ], [ 0.0, -1.68, 0.792, 0.0, -1.724, 0.644, 0.0, 1.74, 0.712, 0.0 ], [ 0.0, 0.0, 0.0, 0.792, 1.364, 0.0, 0.644, 1.188, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.68, 0.792, 1.364, -1.724, 0.644, 1.188, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -1.68, 0.792, 0.0, -1.724, 0.644, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.644 ] ]
    Error: [ [ -1.4653833702027441E-12, 8.550937735662956E-13, 0.0, 7.151390590820483E-12, 2.9054536554440347E-12, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ -7.249756350802272E-13, 2.975508728297882E-12, -2.5512925105886097E-13, 1.2287282302736457E-11, 2.710498492319857E-12, 2.9054536554440347E-12, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 1.4954704141700859E-12, 7.55062679047569E-13, 0.0, -1.035393992765421E-12, -1.730393606180769E-12, 0.0, 0.0, 0.0, 0.0 ], [ -1.6504575484077577E-12, 1.475042310516983E-12, 0.0, 1.1857292925299134E-11, 8.550937735662956E-13, 0.0, -1.730393606180769E-12, 2.9054536554440347E-12, 0.0, 0.0 ], [ 5.402345237826012E-13, 5.699885008425554E-13, 3.6481928589182644E-13, 1.4818146709671964E-11, -1.4653833702027441E-12, 8.550937735662956E-13, -1.035393992765421E-12, 7.151390590820483E-12, 2.9054536554440347E-12, 0.0 ], [ 0.0, 5.402345237826012E-13, 5.699885008425554E-13, 0.0, 1.4954704141700859E-12, 2.975508728297882E-12, 0.0, 3.405498105735205E-12, -1.730393606180769E-12, 0.0 ], [ 0.0, 0.0, 0.0, 5.699885008425554E-13, 1.475042310516983E-12, 0.0, -1.4653833702027441E-12, 8.550937735662956E-13, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.386291081928448E-11, 5.699885008425554E-13, -2.965849787983643E-12, 1.4954704141700859E-12, 2.975508728297882E-12, 8.550937735662956E-13, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 5.402345237826012E-13, 5.699885008425554E-13, 0.0, 5.936362512670712E-12, 2.975508728297882E-12, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4653833702027441E-12 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.1371e-13 +- 1.9175e-12 [0.0000e+00 - 1.5353e-11] (972#)
    relativeTol: 3.6387e-12 +- 1.0729e-11 [6.8674e-16 - 1.0690e-10] (392#)
    
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
...[skipping 879 bytes](etc/2.txt)...
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



