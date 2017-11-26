# ConvolutionLayer
## AsymmetricTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.07 seconds: 
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
      "id": "23c3c755-e008-48dd-8cca-22b000000002",
      "isFrozen": false,
      "name": "ConvolutionLayer/23c3c755-e008-48dd-8cca-22b000000002",
      "filter": {
        "dimensions": [
          3,
          3,
          8
        ],
        "data": [
          -1.068,
          -0.688,
          0.272,
          0.316,
          1.376,
          -1.792,
          0.0,
          -1.16,
          1.08,
          1.12,
          -1.924,
          -1.864,
          -0.324,
          -0.564,
          0.06,
          1.948,
          1.124,
          0.52,
          0.204,
          0.148,
          -1.472,
          0.304,
          0.984,
          -0.064,
          -1.408,
          0.648,
          -1.872,
          0.42,
          -1.884,
          0.94,
          -1.392,
          1.428,
          -1.092,
          -0.048,
          -1.34,
          0.424,
          0.048,
          1.028,
          -1.744,
          0.576,
          -1.58,
          -1.728,
          1.028,
          -1.404,
          0.272,
          -1.972,
          0.42,
          -1.58,
          -1.884,
          -0.972,
          -0.272,
          -1.912,
          0.36,
          1.316,
          -0.428,
          1.244,
          -1.6,
          0.444,
          1.92,
          1.152,
          -1.24,
          0.388,
          0.304,
          0.864,
          0.168,
          -0.844,
          0.784,
          0.628,
          0.456,
          -0.516,
          0.988,
          1.464
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.02 seconds: 
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
    	[ [ 1.06, 1.304 ], [ -0.884, 1.616 ], [ -1.208, 1.264 ] ],
    	[ [ 1.288, -1.244 ], [ 1.748, -1.3 ], [ -0.396, 1.988 ] ],
    	[ [ 1.768, -0.388 ], [ -0.268, -1.212 ], [ 1.896, 1.244 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.485824, 2.760288, 2.64664, -8.283344000000001 ], [ 3.744528, 6.274144, 6.396543999999999, -0.38560000000000005 ], [ 3.963792, 10.267184, 0.155759999999999, 2.8796160000000004 ] ],
    	[ [ -4.969296, -4.2651840000000005, -2.4523040000000003, -5.332064 ], [ -4.290224, -0.15990400000000035, 7.897792000000002, -4.279599999999999 ], [ -0.6710080000000006, 8.150223999999998, 2.7318560000000005, 1.6267360000000004 ] ],
    	[ [ 2.21656, -9.948896, -1.7967199999999997, -4.860576 ], [ -7.347472, -8.075952000000001, -0.7903839999999989, -1.4888480000000002 ], [ 8.262336, 2.3535359999999996, 0.19304000000000013, -0.9442079999999994 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:123](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L123) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "23c3c755-e008-48dd-8cca-22b00000000c",
      "isFrozen": false,
      "name": "ConvolutionLayer/23c3c755-e008-48dd-8cca-22b00000000c",
      "filter": {
        "dimensions": [
          3,
          3,
          8
        ],
        "data": [
          -1.068,
          -0.688,
          0.272,
          0.316,
          1.376,
          -1.792,
          0.0,
          -1.16,
          1.08,
          1.12,
          -1.924,
          -1.864,
          -0.324,
          -0.564,
          0.06,
          1.948,
          1.124,
          0.52,
          0.204,
          0.148,
          -1.472,
          0.304,
          0.984,
          -0.064,
          -1.408,
          0.648,
          -1.872,
          0.42,
          -1.884,
          0.94,
          -1.392,
          1.428,
          -1.092,
          -0.048,
          -1.34,
          0.424,
          0.048,
          1.028,
          -1.744,
          0.576,
          -1.58,
          -1.728,
          1.028,
          -1.404,
          0.272,
          -1.972,
          0.42,
          -1.58,
          -1.884,
          -0.972,
          -0.272,
          -1.912,
          0.36,
          1.316,
          -0.428,
          1.244,
          -1.6,
          0.444,
          1.92,
          1.152,
          -1.24,
          0.388,
          0.304,
          0.864,
          0.168,
          -0.844,
          0.784,
          0.628,
          0.456,
          -0.516,
          0.988,
          1.464
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
    	[ [ 1.06, 1.304 ], [ -0.884, 1.616 ], [ -1.208, 1.264 ] ],
    	[ [ 1.288, -1.244 ], [ 1.748, -1.3 ], [ -0.396, 1.988 ] ],
    	[ [ 1.768, -0.388 ], [ -0.268, -1.212 ], [ 1.896, 1.244 ] ]
    ]]
    Subject Output: [
    	[ [ 1.485824, 2.760288, 2.64664, -8.283344000000001 ], [ 3.744528, 6.274144, 6.396543999999999, -0.38560000000000005 ], [ 3.963792, 10.267184, 0.155759999999999, 2.8796160000000004 ] ],
    	[ [ -4.969296, -4.2651840000000005, -2.4523040000000003, -5.332064 ], [ -4.290224, -0.15990400000000035, 7.897792000000002, -4.279599999999999 ], [ -0.6710080000000006, 8.150223999999998, 2.7318560000000005, 1.6267360000000004 ] ],
    	[ [ 2.21656, -9.948896, -1.7967199999999997, -4.860576 ], [ -7.347472, -8.075952000000001, -0.7903839999999989, -1.4888480000000002 ], [ 8.262336, 2.3535359999999996, 0.19304000000000013, -0.9442079999999994 ] ]
    ]
    Reference Output: [
    	[ [ -0.5711199999999999, 6.961952, 6.1784, 1.112304 ], [ -5.656496, 5.850319999999999, 3.5709280000000008, 0.7876479999999998 ], [ -5.2192, 1.3144639999999999, 1.2054879999999997, 4.546656 ] ],
    	[ [ -5.449552, -2.374944, -1.5003039999999999, -9.136336 ], [ 1.6391520000000004, 11.130240000000002, -0.6969440000000006, 3.072512 ], [ -4.353296000000001, -0.12427199999999937, 10.744095999999997, -0.08289600000000014 ] ],
    	[ [ 4.568448, -1.402048, -2.561215999999999, 3.3488959999999994 ], [ -3.3936799999999994, -2.860415999999999, -6.76416, -13.086208 ], [ 1.4645119999999991, -4.3824, 2.2582879999999994, 2.8272959999999996 ] ]
    ]
    Error: [
    	[ [ 2.0569439999999997, -4.201664, -3.53176, -9.395648000000001 ], [ 9.401024, 0.42382400000000064, 2.825615999999998, -1.1732479999999998 ], [ 9.182992, 8.952720000000001, -1.0497280000000007, -1.6670399999999992 ] ],
    	[ [ 0.4802559999999998, -1.8902400000000004, -0.9520000000000004, 3.804272 ], [ -5.929376, -11.290144000000003, 8.594736000000003, -7.352112 ], [ 3.6822880000000007, 8.274495999999997, -8.012239999999997, 1.7096320000000005 ] ],
    	[ [ -2.351888, -8.546847999999999, 0.7644959999999994, -8.209472 ], [ -3.9537920000000004, -5.215536000000002, 5.973776000000002, 11.597359999999998 ], [ 6.797824, 6.735935999999999, -2.065247999999999, -3.771503999999999 ] ]
    ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=5.0505e+00 +- 3.3299e+00 [4.2382e-01 - 1.1597e+01] (36#), relativeTol=6.7581e-01 +- 3.2776e-01 [3.4956e-02 - 1.0000e+00] (36#)}
    	at com.simiacryptus.mindseye.layers.EquivalencyTester.test(EquivalencyTester.java:66)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.lambda$test$14(LayerTestBase.java:125)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:142)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:77)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:134)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:141)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:123)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:65)
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



