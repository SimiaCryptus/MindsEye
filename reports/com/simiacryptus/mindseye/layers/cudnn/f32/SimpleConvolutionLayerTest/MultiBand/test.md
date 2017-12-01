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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "f4569375-56fe-4e46-925c-95f40000014e",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/f4569375-56fe-4e46-925c-95f40000014e",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          -1.24,
          0.036,
          0.764,
          -1.432,
          -1.96,
          -1.392,
          -1.216,
          -0.528,
          0.052
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
    	[ [ 0.9, -1.624, 0.188 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.0308319330215454, 1.6325440406799316, -0.22715196013450623 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "f4569375-56fe-4e46-925c-95f40000014f",
      "isFrozen": false,
      "name": "ConvolutionLayer/f4569375-56fe-4e46-925c-95f40000014f",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          -1.24,
          -1.432,
          -1.216,
          0.036,
          -1.96,
          -0.528,
          0.764,
          -1.392,
          0.052
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
    	[ [ 0.9, -1.624, 0.188 ] ]
    ]
    Error: [
    	[ [ 6.697845478242925E-8, 4.0679931423426297E-8, 3.986549371171044E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 4.9175e-08 +- 1.2594e-08 [3.9865e-08 - 6.6978e-08] (3#)
    relativeTol: 4.4232e-08 +- 3.1840e-08 [1.2459e-08 - 8.7751e-08] (3#)
    
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
    	[ [ 0.9, -1.624, 0.188 ] ]
    ]
    Output: [
    	[ [ -1.0308319330215454, 1.6325440406799316, -0.22715196013450623 ] ]
    ]
    Measured: [ [ -1.2409687042236328, -1.4317035675048828, -1.2159347534179688 ], [ 0.03695487976074219, -1.958608627319336, -0.527501106262207 ], [ 0.762939453125, -1.392364501953125, 0.05200505256652832 ] ]
    Implemented: [ [ -1.2400000095367432, -1.4320000410079956, -1.215999960899353 ], [ 0.035999998450279236, -1.9600000381469727, -0.527999997138977 ], [ 0.7639999985694885, -1.3919999599456787, 0.052000001072883606 ] ]
    Error: [ [ -9.686946868896484E-4, 2.9647350311279297E-4, 6.520748138427734E-5 ], [ 9.548813104629517E-4, 0.0013914108276367188, 4.988908767700195E-4 ], [ -0.0010605454444885254, -3.6454200744628906E-4, 5.0514936447143555E-6 ] ]
    Learning Gradient for weight set 0
    Inputs: [
    	[ [ 0.9, -1.624, 0.188 ] ]
    ]
    Outputs: [
    	[ [ -1.0308319330215454, 1.6325440406799316, -0.22715196013450623 ] ]
    ]
    Measured Gradient: [ [ 0.9000301361083984, 0.0, 0.0 ], [ -1.6236305236816406, 0.0, 0.0 ], [ 0.18715858459472656, 0.0, 0.0 ], [ 0.0, 0.9000301361083984, 0.0 ], [ 0.0, -1.6248226165771484, 0.0 ], [ 0.0, 0.18835067749023438, 0.0 ], [ 0.0, 0.0, 0.9000301361083984 ], [ 0.0, 0.0, -1.6242265701293945 ], [ 0.0, 0.0, 0.18805265426635742 ] ]
    Implemented Gradient: [ [ 0.8999999761581421, 0.0, 0.0 ], [ -1.6239999532699585, 0.0, 0.0 ], [ 0.18799999356269836, 0.0, 0.0 ], [ 0.0, 0.8999999761581421, 0.0 ], [ 0.0, -1.6239999532699585, 0.0 ], [ 0.0, 0.18799999356269836, 0.0 ], [ 0.0, 0.0, 0.8999999761581421 ], [ 0.0, 0.0, -1.6239999532699585 ], [ 0.0, 0.0, 0.18799999356269836 ] ]
    Error: [ [ 3.0159950256347656E-5, 0.0, 0.0 ], [ 3.694295883178711E-4, 0.0, 0.0 ], [ -8.414089679718018E-4, 0.0, 0.0 ], [ 0.0, 3.0159950256347656E-5, 0.0 ], [ 0.0, -8.226633071899414E-4, 0.0 ], [ 0.0, 3.5068392753601074E-4, 0.0 ], [ 0.0, 0.0, 3.0159950256347656E-5 ], [ 0.0, 0.0, -2.2661685943603516E-4 ], [ 0.0, 0.0, 5.266070365905762E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.3221e-04 +- 3.7803e-04 [0.0000e+00 - 1.3914e-03] (36#)
    relativeTol: 1.0618e-03 +- 2.9631e-03 [1.6755e-05 - 1.3089e-02] (18#)
    
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
...[skipping 796 bytes](etc/1.txt)...
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



