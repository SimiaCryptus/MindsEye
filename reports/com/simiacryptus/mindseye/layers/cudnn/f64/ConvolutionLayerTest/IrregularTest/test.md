# ConvolutionLayer
## IrregularTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.05 seconds: 
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
      "id": "f68b7be3-1b67-45e0-9eb0-70cb00000002",
      "isFrozen": false,
      "name": "ConvolutionLayer/f68b7be3-1b67-45e0-9eb0-70cb00000002",
      "filter": {
        "dimensions": [
          3,
          3,
          6
        ],
        "data": [
          0.004,
          -1.584,
          1.216,
          -0.02,
          1.488,
          1.464,
          -1.704,
          1.612,
          -1.084,
          -1.252,
          -0.168,
          -0.316,
          -1.704,
          -1.04,
          -1.644,
          1.328,
          -1.128,
          0.852,
          1.66,
          1.448,
          1.056,
          -0.272,
          0.08,
          0.46,
          -1.184,
          1.468,
          -1.128,
          0.38,
          0.34,
          -1.32,
          1.324,
          -1.204,
          0.436,
          1.892,
          0.008,
          0.352,
          0.316,
          0.24,
          1.44,
          0.388,
          -1.364,
          -0.3,
          0.86,
          -1.708,
          -1.612,
          0.276,
          0.204,
          -1.78,
          -1.4,
          0.376,
          -0.12,
          0.484,
          0.972,
          -0.448
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
    	[ [ 0.76, -1.712 ], [ -0.584, 1.408 ], [ 0.476, -1.456 ] ],
    	[ [ 0.564, 1.848 ], [ -1.588, 0.712 ], [ -0.62, 0.068 ] ],
    	[ [ -0.708, 0.052 ], [ 0.848, -1.192 ], [ 0.196, 0.12 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 4.6193919999999995, -3.9562559999999998, -3.717472 ], [ -8.107312, 1.152896, -1.9152480000000005 ], [ 3.5741919999999996, -0.502352, -3.332624000000001 ] ],
    	[ [ 3.659952, -0.4942720000000004, -2.5943840000000002 ], [ 3.9899839999999998, -3.7282239999999995, 7.942559999999999 ], [ -3.430048, 6.093888, 2.116304 ] ],
    	[ [ -3.6219520000000003, -4.403616, 1.7700800000000003 ], [ -5.410544, 0.31740799999999986, -2.0430080000000004 ], [ -0.0397759999999994, 2.744736, -2.499136 ] ]
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
      "id": "f68b7be3-1b67-45e0-9eb0-70cb0000000c",
      "isFrozen": false,
      "name": "ConvolutionLayer/f68b7be3-1b67-45e0-9eb0-70cb0000000c",
      "filter": {
        "dimensions": [
          3,
          3,
          6
        ],
        "data": [
          0.004,
          -1.584,
          1.216,
          -0.02,
          1.488,
          1.464,
          -1.704,
          1.612,
          -1.084,
          -1.252,
          -0.168,
          -0.316,
          -1.704,
          -1.04,
          -1.644,
          1.328,
          -1.128,
          0.852,
          1.66,
          1.448,
          1.056,
          -0.272,
          0.08,
          0.46,
          -1.184,
          1.468,
          -1.128,
          0.38,
          0.34,
          -1.32,
          1.324,
          -1.204,
          0.436,
          1.892,
          0.008,
          0.352,
          0.316,
          0.24,
          1.44,
          0.388,
          -1.364,
          -0.3,
          0.86,
          -1.708,
          -1.612,
          0.276,
          0.204,
          -1.78,
          -1.4,
          0.376,
          -0.12,
          0.484,
          0.972,
          -0.448
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
    	[ [ 0.76, -1.712 ], [ -0.584, 1.408 ], [ 0.476, -1.456 ] ],
    	[ [ 0.564, 1.848 ], [ -1.588, 0.712 ], [ -0.62, 0.068 ] ],
    	[ [ -0.708, 0.052 ], [ 0.848, -1.192 ], [ 0.196, 0.12 ] ]
    ]]
    Subject Output: [
    	[ [ 4.6193919999999995, -3.9562559999999998, -3.717472 ], [ -8.107312, 1.152896, -1.9152480000000005 ], [ 3.5741919999999996, -0.502352, -3.332624000000001 ] ],
    	[ [ 3.659952, -0.4942720000000004, -2.5943840000000002 ], [ 3.9899839999999998, -3.7282239999999995, 7.942559999999999 ], [ -3.430048, 6.093888, 2.116304 ] ],
    	[ [ -3.6219520000000003, -4.403616, 1.7700800000000003 ], [ -5.410544, 0.31740799999999986, -2.0430080000000004 ], [ -0.0397759999999994, 2.744736, -2.499136 ] ]
    ]
    Reference Output: [
    	[ [ 7.295584000000001, 3.949936, -6.321488 ], [ 0.9313440000000005, 6.442464000000001, -1.0221119999999995 ], [ 5.686655999999999, -0.6688480000000002, 2.300128 ] ],
    	[ [ -1.1973440000000002, -1.4051040000000004, -2.7836480000000003 ], [ -0.7018240000000002, -4.304303999999999, 7.107487999999999 ], [ -5.913168, -2.172192, -3.0437760000000003 ] ],
    	[ [ -4.104048, 0.28232, -1.95896 ], [ -1.532512, 1.642112, -4.11664 ], [ 2.598592, -0.7899680000000002, 1.3259679999999996 ] ]
    ]
    Error: [
    	[ [ -2.6761920000000012, -7.906192, 2.6040160000000006 ], [ -9.038656000000001, -5.289568000000001, -0.893136000000001 ], [ -2.1124639999999997, 0.1664960000000002, -5.632752000000001 ] ],
    	[ [ 4.857296, 0.910832, 0.1892640000000001 ], [ 4.691808, 0.5760799999999997, 0.8350720000000003 ], [ 2.4831199999999995, 8.266079999999999, 5.160080000000001 ] ],
    	[ [ 0.4820959999999994, -4.685936000000001, 3.7290400000000004 ], [ -3.8780319999999997, -1.324704, 2.073632 ], [ -2.6383679999999994, 3.5347040000000005, -3.8251039999999996 ] ]
    ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=3.3504e+00 +- 2.4340e+00 [1.6650e-01 - 9.0387e+00] (27#), relativeTol=6.0726e-01 +- 3.8524e-01 [3.5192e-02 - 1.0000e+00] (27#)}
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



