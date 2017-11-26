# ConvolutionLayer
## IrregularTest
### Json Serialization
Code from [LayerTestBase.java:75](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L75) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer",
      "id": "055e4cd6-0193-4699-9154-1c1700000014",
      "isFrozen": false,
      "name": "ConvolutionLayer/055e4cd6-0193-4699-9154-1c1700000014",
      "filter": {
        "dimensions": [
          3,
          3,
          10
        ],
        "data": [
          -4.896,
          1.3279999999999998,
          0.6240000000000001,
          -2.112,
          -1.52,
          -1.8719999999999999,
          -2.488,
          0.488,
          -0.704,
          1.2879999999999998,
          -2.8,
          -4.968,
          2.44,
          -4.5920000000000005,
          -2.128,
          -2.016,
          2.352,
          1.8319999999999999,
          -2.48,
          -2.072,
          2.976,
          -4.888,
          2.504,
          -3.704,
          -3.464,
          -0.16000000000000003,
          1.7999999999999998,
          -0.43999999999999995,
          1.7440000000000002,
          -3.704,
          -4.192,
          -2.136,
          2.32,
          -0.648,
          2.112,
          -4.464,
          -0.744,
          -2.2560000000000002,
          -3.672,
          -0.488,
          2.472,
          2.424,
          -0.6639999999999999,
          0.48,
          -4.4079999999999995,
          1.1760000000000002,
          -0.896,
          -3.512,
          -2.6799999999999997,
          1.8239999999999998,
          2.072,
          1.12,
          -1.3519999999999999,
          -2.944,
          2.48,
          -4.064,
          -3.168,
          -0.84,
          -0.5920000000000001,
          -1.3599999999999999,
          0.6719999999999999,
          -1.752,
          0.3999999999999999,
          -2.0,
          -3.0,
          -0.728,
          2.136,
          -3.688,
          0.8799999999999999,
          -1.3439999999999999,
          -0.17600000000000005,
          -3.328,
          -1.384,
          2.712,
          -1.144,
          0.6080000000000001,
          -2.56,
          1.936,
          1.608,
          1.88,
          -2.992,
          -1.896,
          2.48,
          0.752,
          2.248,
          -4.2,
          -3.96,
          -1.296,
          1.616,
          1.6
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:112](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L112) executed in 0.01 seconds: 
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
    	[ [ -1.04, 1.148 ], [ -0.064, 1.468 ], [ -1.528, 1.9 ] ],
    	[ [ -1.948, 1.112 ], [ 0.444, -0.396 ], [ -1.812, -1.448 ] ],
    	[ [ -1.824, -0.868 ], [ 1.528, 0.0 ], [ -1.452, 0.256 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -1.1843202114105225, -10.367938041687012, -6.488832473754883, 7.413247585296631, 1.048223853111267, 5.227168560028076 ], [ 11.571135520935059, -7.898399353027344, -20.72393798828125, -4.499262809753418, 13.009183883666992, 12.669248580932617 ], [ 17.985891342163086, -5.965025901794434, -4.603230953216553, 15.354816436767578, -7.170238494873047, 9.874176979064941 ] ],
    	[ [ 9.793057441711426, -12.881537437438965, -14.824352264404297, 11.665791511535645, -13.014336585998535, 4.4482879638671875 ], [ 10.772575378417969, 12.502143859863281, -4.863713264465332, -11.263681411743164, 15.238846778869629, -3.825249195098877 ], [ -4.067615509033203, -8.654752731323242, 9.26956844329834, 3.4687037467956543, 1.1597113609313965, -3.0192012786865234 ] ],
    	[ [ 1.2545602321624756, -14.052350997924805, 8.12095832824707, 2.508544445037842, -0.9502730369567871, 2.8073272705078125 ], [ -6.963809013366699, 28.545085906982422, 4.797506332397461, 8.72451400756836, 8.725215911865234, -4.955711841583252 ], [ 11.323808670043945, -7.546015739440918, 2.9335997104644775, -1.2247366905212402, 6.039134979248047, 2.4990077018737793 ] ]
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
      "id": "055e4cd6-0193-4699-9154-1c1700000020",
      "isFrozen": false,
      "name": "ConvolutionLayer/055e4cd6-0193-4699-9154-1c1700000020",
      "filter": {
        "dimensions": [
          3,
          3,
          10
        ],
        "data": [
          -4.896,
          1.3279999999999998,
          0.6240000000000001,
          -2.112,
          -1.52,
          -1.8719999999999999,
          -2.488,
          0.488,
          -0.704,
          1.2879999999999998,
          -2.8,
          -4.968,
          2.44,
          -4.5920000000000005,
          -2.128,
          -2.016,
          2.352,
          1.8319999999999999,
          -2.48,
          -2.072,
          2.976,
          -4.888,
          2.504,
          -3.704,
          -3.464,
          -0.16000000000000003,
          1.7999999999999998,
          -0.43999999999999995,
          1.7440000000000002,
          -3.704,
          -4.192,
          -2.136,
          2.32,
          -0.648,
          2.112,
          -4.464,
          -0.744,
          -2.2560000000000002,
          -3.672,
          -0.488,
          2.472,
          2.424,
          -0.6639999999999999,
          0.48,
          -4.4079999999999995,
          1.1760000000000002,
          -0.896,
          -3.512,
          -2.6799999999999997,
          1.8239999999999998,
          2.072,
          1.12,
          -1.3519999999999999,
          -2.944,
          2.48,
          -4.064,
          -3.168,
          -0.84,
          -0.5920000000000001,
          -1.3599999999999999,
          0.6719999999999999,
          -1.752,
          0.3999999999999999,
          -2.0,
          -3.0,
          -0.728,
          2.136,
          -3.688,
          0.8799999999999999,
          -1.3439999999999999,
          -0.17600000000000005,
          -3.328,
          -1.384,
          2.712,
          -1.144,
          0.6080000000000001,
          -2.56,
          1.936,
          1.608,
          1.88,
          -2.992,
          -1.896,
          2.48,
          0.752,
          2.248,
          -4.2,
          -3.96,
          -1.296,
          1.616,
          1.6
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

Returns: 

```
    java.lang.AssertionError
    	at com.simiacryptus.mindseye.lang.Tensor.minus(Tensor.java:725)
    	at com.simiacryptus.mindseye.layers.EquivalencyTester.test(EquivalencyTester.java:61)
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
    	at org.junit.runners.Suite.runChild(Suite.java:128)
    	at org.junit.runners.Suite.runChild(Suite.java:27)
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



