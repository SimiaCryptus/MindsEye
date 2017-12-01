# SimpleConvolutionLayer
## MultiBand
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
      "id": "f4569375-56fe-4e46-925c-95f400000951",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/f4569375-56fe-4e46-925c-95f400000951",
      "filter": {
        "dimensions": [
          3,
          3,
          9
        ],
        "data": [
          1.676,
          -0.156,
          0.908,
          1.016,
          -0.432,
          -1.6,
          1.364,
          0.408,
          -0.508,
          -1.508,
          1.776,
          1.68,
          1.812,
          0.896,
          1.712,
          1.488,
          1.312,
          1.324,
          -0.74,
          1.7,
          1.012,
          -1.036,
          -0.968,
          1.264,
          -1.06,
          0.456,
          0.92,
          -0.292,
          0.692,
          -1.844,
          -0.708,
          -0.224,
          0.084,
          -1.34,
          -0.2,
          1.988,
          -1.12,
          -1.26,
          -0.404,
          0.348,
          -1.54,
          0.208,
          -1.264,
          0.284,
          -1.248,
          0.08,
          -1.796,
          0.444,
          -1.32,
          1.668,
          -0.24,
          0.588,
          1.22,
          -1.4,
          1.484,
          1.636,
          -1.584,
          -0.704,
          0.548,
          1.32,
          0.812,
          -1.46,
          -0.536,
          -0.692,
          -1.296,
          0.764,
          0.768,
          1.36,
          -1.3,
          1.688,
          -1.772,
          -0.796,
          0.232,
          -0.34,
          -1.512,
          -1.648,
          0.944,
          -0.284,
          1.8,
          -1.972,
          -1.176
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    	[ [ -0.568, -1.316, 0.572 ], [ 0.32, 1.06, -1.348 ], [ -0.216, -0.524, -1.68 ] ],
    	[ [ 1.348, -1.724, -0.368 ], [ -0.592, -0.972, 1.328 ], [ -1.084, -0.824, -1.3 ] ],
    	[ [ 1.0, -0.96, -1.788 ], [ -0.48, -0.332, -0.024 ], [ -0.128, -0.676, 1.608 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -3.8286240000000005, 4.713984, -3.5173439999999996 ], [ -7.021343999999999, -0.3670400000000008, -3.061632 ], [ -2.7524960000000003, 1.6450559999999996, 0.4374719999999996 ] ],
    	[ [ -0.21835199999999946, -0.07824000000000014, 2.04328 ], [ -11.008704000000002, 5.407888000000001, 0.4913439999999994 ], [ -6.7412800000000015, 1.0639840000000003, -5.603231999999999 ] ],
    	[ [ -6.516928000000001, 0.3182080000000001, -0.4800639999999999 ], [ -6.2091199999999995, 2.3305279999999997, 7.39048 ], [ -3.833232, 1.9508, 1.4036959999999996 ] ]
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
      "id": "f4569375-56fe-4e46-925c-95f400000952",
      "isFrozen": false,
      "name": "ConvolutionLayer/f4569375-56fe-4e46-925c-95f400000952",
      "filter": {
        "dimensions": [
          3,
          3,
          9
        ],
        "data": [
          1.676,
          -0.156,
          0.908,
          1.016,
          -0.432,
          -1.6,
          1.364,
          0.408,
          -0.508,
          -0.292,
          0.692,
          -1.844,
          -0.708,
          -0.224,
          0.084,
          -1.34,
          -0.2,
          1.988,
          1.484,
          1.636,
          -1.584,
          -0.704,
          0.548,
          1.32,
          0.812,
          -1.46,
          -0.536,
          -1.508,
          1.776,
          1.68,
          1.812,
          0.896,
          1.712,
          1.488,
          1.312,
          1.324,
          -1.12,
          -1.26,
          -0.404,
          0.348,
          -1.54,
          0.208,
          -1.264,
          0.284,
          -1.248,
          -0.692,
          -1.296,
          0.764,
          0.768,
          1.36,
          -1.3,
          1.688,
          -1.772,
          -0.796,
          -0.74,
          1.7,
          1.012,
          -1.036,
          -0.968,
          1.264,
          -1.06,
          0.456,
          0.92,
          0.08,
          -1.796,
          0.444,
          -1.32,
          1.668,
          -0.24,
          0.588,
          1.22,
          -1.4,
          0.232,
          -0.34,
          -1.512,
          -1.648,
          0.944,
          -0.284,
          1.8,
          -1.972,
          -1.176
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
    	[ [ -0.568, -1.316, 0.572 ], [ 0.32, 1.06, -1.348 ], [ -0.216, -0.524, -1.68 ] ],
    	[ [ 1.348, -1.724, -0.368 ], [ -0.592, -0.972, 1.328 ], [ -1.084, -0.824, -1.3 ] ],
    	[ [ 1.0, -0.96, -1.788 ], [ -0.48, -0.332, -0.024 ], [ -0.128, -0.676, 1.608 ] ]
    ]
    Error: [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (27#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (27#)
    
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
    	[ [ -0.568, -1.316, 0.572 ], [ 0.32, 1.06, -1.348 ], [ -0.216, -0.524, -1.68 ] ],
    	[ [ 1.348, -1.724, -0.368 ], [ -0.592, -0.972, 1.328 ], [ -1.084, -0.824, -1.3 ] ],
    	[ [ 1.0, -0.96, -1.788 ], [ -0.48, -0.332, -0.024 ], [ -0.128, -0.676, 1.608 ] ]
    ]
    Output: [
    	[ [ -3.8286240000000005, 4.713984, -3.5173439999999996 ], [ -7.021343999999999, -0.3670400000000008, -3.061632 ], [ -2.7524960000000003, 1.6450559999999996, 0.4374719999999996 ] ],
    	[ [ -0.21835199999999946, -0.07824000000000014, 2.04328 ], [ -11.008704000000002, 5.407888000000001, 0.4913439999999994 ], [ -6.7412800000000015, 1.0639840000000003, -5.603231999999999 ] ],
    	[ [ -6.516928000000001, 0.3182080000000001, -0.4800639999999999 ], [ -6.2091199999999995, 2.3305279999999997, 7.39048 ], [ -3.833232, 1.9508, 1.4036959999999996 ] ]
    ]
    Measured: [ [ -0.43199999999909977, -1.6000000000016001, 0.0, 0.40799999998952785, -0.5080000000035056, 0.0, 0.0, 0.0, 0.0, -0.22399999999755948 ], [ 1.0160000000025704, -0.43200000000354066, -1.5999999999927184, 1.3639999999881525, 0.4080000000072914, -0.5080000000035056, 0.0, 0.0, 0.0, -0.7079999999959341 ], [ 0.0, 1.0159999999981295, -0.4319999999857771, 0.0, 1.364000000005916, 0.40799999998952785, 0.0, 0.0, 0.0, 0.0 ], [ -0.15599999999782455, 0.9079999999972443, 0.0, -0.43200000000354066, -1.5999999999927184, 0.0, 0.40799999999840963, -0.5080000000035056, 0.0, 0.6920000000043558 ], [ 1.6760000000015651, -0.15600000000226544, 0.9080000000061261, 1.0159999999892477, -0.43200000000354066, -1.600000000010482, 1.3639999999970343, 0.40799999999840963, -0.5079999999946239, -0.2920000000017353 ], [ 0.0, 1.6759999999926833, -0.15599999999338365, 0.0, 1.0160000000070113, -0.43200000000354066, 0.0, 1.3639999999970343, 0.40799999999840963, 0.0 ], [ 0.0, 0.0, 0.0, -0.15599999999338365, 0.9080000000061261, 0.0, -0.43200000000354066, -1.6000000000016001, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 1.6759999999926833, -0.15599999999338365, 0.9079999999883626, 1.0159999999981295, -0.43200000000354066, -1.600000000001600
```
...[skipping 5244 bytes](etc/1.txt)...
```
    1.348, 0.0, 0.32, -0.592, 0.0, -0.216, -1.084, 0.0 ], [ 0.0, 0.0, 0.0, 1.348, 1.0, 0.0, -0.592, -0.48, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -0.568, 1.348, 1.0, 0.32, -0.592, -0.48, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.568, 1.348, 0.0, 0.32, -0.592, 0.0 ], [ -0.972, -0.332, 0.0, -0.824, -0.676, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Error: [ [ 7.401856905175919E-13, -4.920952534348544E-12, 0.0, -2.305267088331675E-12, -3.6807223935397815E-12, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 3.2002178684820137E-13, -8.14159850648366E-12, 3.9608316626527085E-12, -1.770333879491659E-12, -2.305267088331675E-12, -3.6807223935397815E-12, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 3.2002178684820137E-13, 9.621969887518844E-12, 0.0, -1.770333879491659E-12, -2.305267088331675E-12, 0.0, 0.0, 0.0, 0.0 ], [ 1.0149658891123181E-12, -2.3305801732931286E-12, 0.0, -8.14159850648366E-12, 3.9608316626527085E-12, 0.0, -2.305267088331675E-12, -3.6807223935397815E-12, 0.0, 0.0 ], [ 1.4303003226245892E-12, -3.425926209388308E-12, 1.5432988220709376E-11, -8.561762410153051E-12, -8.14159850648366E-12, -1.3802736731349796E-11, -1.770333879491659E-12, -2.305267088331675E-12, 5.201061803461471E-12, 0.0 ], [ 0.0, -3.010591775876037E-12, 1.4337642184614197E-11, 0.0, -8.561762410153051E-12, -8.14159850648366E-12, 0.0, -1.770333879491659E-12, -2.305267088331675E-12, 0.0 ], [ 0.0, 0.0, 0.0, -3.425926209388308E-12, -2.3305801732931286E-12, 0.0, 7.401856905175919E-13, 3.9608316626527085E-12, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, -1.189237597287729E-11, -3.425926209388308E-12, -2.3305801732931286E-12, 3.2002178684820137E-13, 7.401856905175919E-13, 3.9608316626527085E-12, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 5.871192421125215E-12, -3.425926209388308E-12, 0.0, 3.2002178684820137E-13, 7.401856905175919E-13, 0.0 ], [ 9.15267861500979E-13, -7.32641725065264E-12, 0.0, -1.4902523659543476E-12, 4.9861226258940405E-12, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 8.6434e-13 +- 1.9683e-12 [0.0000e+00 - 1.6063e-11] (2916#)
    relativeTol: 3.6534e-12 +- 1.3550e-11 [1.9165e-14 - 2.9193e-10] (882#)
    
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
    	at org.junit.runners
```
...[skipping 796 bytes](etc/2.txt)...
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
    	... 35 more
    Caused by: java.util.concurrent.ExecutionException: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at java.util.concurrent.FutureTask.report(FutureTask.java:122)
    	at java.util.concurrent.FutureTask.get(FutureTask.java:192)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$run$8(GpuController.java:213)
    	... 43 more
    Caused by: java.lang.AssertionError: Nonfrozen component not listed in delta. Deltas: []
    	at com.simiacryptus.mindseye.layers.DerivativeTester.lambda$testUnFrozen$17(DerivativeTester.java:142)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$null$7(GpuController.java:213)
    	at java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:511)
    	at java.util.concurrent.FutureTask.run(FutureTask.java:266)
    	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    	at java.lang.Thread.run(Thread.java:748)
    
```



