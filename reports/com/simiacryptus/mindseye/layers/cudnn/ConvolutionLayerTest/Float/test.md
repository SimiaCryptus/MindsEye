# ConvolutionLayer
## Float
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.ConvolutionLayer",
      "id": "17594935-f0c4-482c-8e13-1ce3eeaa545a",
      "isFrozen": false,
      "name": "ConvolutionLayer/17594935-f0c4-482c-8e13-1ce3eeaa545a",
      "filter": [
        [
          [
            -1.016
          ]
        ],
        [
          [
            -1.952
          ]
        ],
        [
          [
            -0.084
          ]
        ],
        [
          [
            -0.988
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ -0.296, -1.4 ], [ -1.284, -0.968 ], [ -0.044, -1.628 ], [ 1.432, 0.7 ], [ 1.472, -0.512 ] ],
    	[ [ -0.32, 0.336 ], [ 0.056, 0.836 ], [ -0.792, 1.184 ], [ -0.96, 1.568 ], [ -1.748, -1.904 ] ],
    	[ [ 0.944, 1.576 ], [ 1.052, 0.18 ], [ -0.984, -1.972 ], [ 1.54, -0.164 ], [ 0.912, -1.956 ] ],
    	[ [ -1.024, 0.932 ], [ -1.364, 0.468 ], [ -0.06, -1.844 ], [ -1.296, -0.564 ], [ -1.232, 1.252 ] ],
    	[ [ 0.276, -0.164 ], [ -0.976, -1.728 ], [ -1.488, 0.476 ], [ -0.084, -1.484 ], [ -0.192, -1.816 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Reference Implementation
Code from [StandardLayerTests.java:93](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L93) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "9ee09c7a-774d-46c0-89d9-85a407a1ef6a",
      "isFrozen": false,
      "name": "ConvolutionLayer/9ee09c7a-774d-46c0-89d9-85a407a1ef6a",
      "filter": [
        [
          [
            -1.016
          ]
        ],
        [
          [
            -1.952
          ]
        ],
        [
          [
            -0.084
          ]
        ],
        [
          [
            -0.988
          ]
        ]
      ],
      "skip": [
        [
          0.0
        ]
      ],
      "simple": true
    }
    Inputs: Optional[[
    	[ [ 0.088, 1.468 ], [ 0.3, 0.852 ], [ 0.16, 0.348 ], [ 1.984, -1.616 ], [ -0.388, 1.132 ] ],
    	[ [ -1.476, -1.792 ], [ -0.66, 1.916 ], [ 1.088, 1.488 ], [ -0.652, 0.988 ], [ 1.172, 0.856 ] ],
    	[ [ 0.044, -1.656 ], [ -1.74, 0.524 ], [ -1.764, -1.816 ], [ -1.864, 0.748 ], [ -0.68, -1.948 ] ],
    	[ [ -1.1, 1.332 ], [ -1.4, 0.208 ], [ -0.872, -1.024 ], [ -0.768, 0.184 ], [ -1.828, 0.78 ] ],
    	[ [ 1.028, 1.008 ], [ 0.252, -0.28 ], [ -0.596, 1.852 ], [ 0.664, -1.58 ], [ -0.572, 0.432 ] ]
    ]]
    Subject Output: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0
```
...[skipping 1150 bytes](etc/19.txt)...
```
    9999999999, -0.6663840000000001 ], [ -0.5419039999999999, 0.2649120000000001 ], [ 0.544864, 0.689728 ] ]
    ]
    Error: [
    	[ [ 0.21272000000000002, 1.62216 ], [ 0.37636800000000004, 1.427376 ], [ 0.19179200000000002, 0.656144 ], [ 1.8800000000000001, 2.2761599999999995 ], [ -0.29912, 0.36103999999999986 ] ],
    	[ [ -1.650144, -4.651648 ], [ -0.5096160000000001, 0.6046879999999998 ], [ 1.2304000000000002, 3.59392 ], [ -0.5794400000000001, -0.2965600000000001 ], [ 1.262656, 3.133472 ] ],
    	[ [ -0.0944, -1.5502399999999998 ], [ -1.723824, -2.878768 ], [ -1.944768, -5.237536 ], [ -1.8309920000000002, -2.899504 ], [ -0.854512, -3.251984 ] ],
    	[ [ -1.0057120000000002, -0.8311840000000001 ], [ -1.404928, -2.5272959999999998 ], [ -0.9719679999999999, -2.713856 ], [ -0.764832, -1.317344 ], [ -1.791728, -2.7976159999999997 ] ],
    	[ [ 1.1291200000000001, 3.00256 ], [ 0.23251199999999997, 0.21526399999999998 ], [ -0.4499679999999999, 0.6663840000000001 ], [ 0.5419039999999999, -0.2649120000000001 ], [ -0.544864, -0.689728 ] ]
    ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=1.4589e+00 +- 1.1912e+00 [9.4400e-02 - 5.2375e+00] (50#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (50#)}
    	at com.simiacryptus.mindseye.test.EquivalencyTester.test(EquivalencyTester.java:66)
    	at com.simiacryptus.mindseye.test.StandardLayerTests.lambda$test$8(StandardLayerTests.java:95)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:157)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:82)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.test.StandardLay
```
...[skipping 1254 bytes](etc/20.txt)...
```
    unner.java:268)
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



