# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.05 seconds: 
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
      "id": "c029c99e-5051-4fad-a581-4a8d00000001",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/c029c99e-5051-4fad-a581-4a8d00000001",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          0.324,
          -1.852,
          0.508,
          1.068,
          -0.3,
          0.568,
          0.5,
          -0.704,
          -0.136,
          -1.512,
          -1.628,
          -0.668,
          -1.484,
          -1.056,
          -0.004,
          0.36,
          -1.464,
          -0.216,
          1.552,
          -1.592,
          -1.5,
          -0.948,
          -1.088,
          0.244,
          -0.28,
          -0.448,
          -1.192,
          0.012,
          -1.716,
          0.116,
          0.58,
          1.348,
          -1.612,
          -0.444,
          1.648,
          -1.116
        ]
      },
      "simple": false,
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.02 seconds: 
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
    	[ [ -1.668, -0.736 ], [ -0.392, 0.532 ], [ 0.688, 0.592 ] ],
    	[ [ 1.896, 1.896 ], [ -0.184, 0.796 ], [ -1.056, 0.112 ] ],
    	[ [ 1.848, -1.552 ], [ 1.804, 1.272 ], [ -0.256, -1.056 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.914400577545166, -0.43993616104125977 ], [ -0.6890726089477539, -3.8073606491088867 ], [ -2.433887481689453, 1.865968108177185 ] ],
    	[ [ -2.087024211883545, 1.012111783027649 ], [ -1.089406967163086, 4.719663619995117 ], [ 2.1431519985198975, 0.3415839672088623 ] ],
    	[ [ -3.883167028427124, -11.382831573486328 ], [ -0.10636746883392334, -5.5210723876953125 ], [ -2.6874561309814453, -0.964112401008606 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "c029c99e-5051-4fad-a581-4a8d00000003",
      "isFrozen": false,
      "name": "ConvolutionLayer/c029c99e-5051-4fad-a581-4a8d00000003",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          0.324,
          -1.852,
          0.508,
          1.068,
          -0.3,
          0.568,
          0.5,
          -0.704,
          -0.136,
          -1.512,
          -1.628,
          -0.668,
          -1.484,
          -1.056,
          -0.004,
          0.36,
          -1.464,
          -0.216,
          1.552,
          -1.592,
          -1.5,
          -0.948,
          -1.088,
          0.244,
          -0.28,
          -0.448,
          -1.192,
          0.012,
          -1.716,
          0.116,
          0.58,
          1.348,
          -1.612,
          -0.444,
          1.648,
          -1.116
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
    Inputs: Optional[[
    	[ [ -1.668, -0.736 ], [ -0.392, 0.532 ], [ 0.688, 0.592 ] ],
    	[ [ 1.896, 1.896 ], [ -0.184, 0.796 ], [ -1.056, 0.112 ] ],
    	[ [ 1.848, -1.552 ], [ 1.804, 1.272 ], [ -0.256, -1.056 ] ]
    ]]
    Subject Output: [
    	[ [ -0.914400577545166, -0.43993616104125977 ], [ -0.6890726089477539, -3.8073606491088867 ], [ -2.433887481689453, 1.865968108177185 ] ],
    	[ [ -2.087024211883545, 1.012111783027649 ], [ -1.089406967163086, 4.719663619995117 ], [ 2.1431519985198975, 0.3415839672088623 ] ],
    	[ [ -3.883167028427124, -11.382831573486328 ], [ -0.10636746883392334, -5.5210723876953125 ], [ -2.6874561309814453, -0.964112401008606 ] ]
    ]
    Reference Output: [
    	[ [ 2.5835040000000005, -0.9316800000000001 ], [ -1.8761760000000003, 2.397664 ], [ -2.3617280000000003, 2.734512 ] ],
    	[ [ 0.3213440000000003, -5.351056000000001 ], [ -0.3872959999999998, 2.86728 ], [ 1.195792, 1.233296 ] ],
    	[ [ -3.979776000000001, -12.011936 ], [ -3.5081919999999993, -6.31416 ], [ -2.110432, -2.722864000000001 ] ]
    ]
    Error: [
    	[ [ -3.4979045775451665, 0.4917438389587403 ], [ 1.1871033910522464, -6.2050246491088865 ], [ -0.07215948168945285, -0.868543891822815 ] ],
    	[ [ -2.408368211883545, 6.36316778302765 ], [ -0.7021109671630861, 1.8523836199951171 ], [ 0.9473599985198975, -0.8917120327911376 ] ],
    	[ [ 0.09660897157287707, 0.6291044265136723 ], [ 3.401824531166076, 0.7930876123046877 ], [ -0.5770241309814454, 1.7587515989913949 ] ]
    ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=1.8191e+00 +- 1.8488e+00 [7.2159e-02 - 6.3632e+00] (18#), relativeTol=4.5774e-01 +- 3.6720e-01 [1.2287e-02 - 1.0000e+00] (18#)}
    	at com.simiacryptus.mindseye.layers.EquivalencyTester.test(EquivalencyTester.java:69)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$14(LayerTestBase.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:157)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:82)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:132)
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
    	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
    	at org.junit.runner.JUnitCore.run(JUnitCore.java:137)
    	at com.intellij.junit4.JUnit4IdeaTestRunner.startRunnerWithArgs(JUnit4IdeaTestRunner.java:68)
    	at com.intellij.rt.execution.junit.IdeaTestRunner$Repeater.startRunnerWithArgs(IdeaTestRunner.java:47)
    	at com.intellij.rt.execution.junit.JUnitStarter.prepareStreamsAndStart(JUnitStarter.java:242)
    	at com.intellij.rt.execution.junit.JUnitStarter.main(JUnitStarter.java:70)
    
```



