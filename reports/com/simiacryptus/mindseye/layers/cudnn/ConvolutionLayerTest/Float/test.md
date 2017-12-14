# ConvolutionLayer
## Float
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "id": "8a8d81e5-1503-4ae7-8465-c62fd81a3dad",
      "isFrozen": false,
      "name": "ConvolutionLayer/8a8d81e5-1503-4ae7-8465-c62fd81a3dad",
      "filter": [
        [
          [
            0.08
          ]
        ],
        [
          [
            1.404
          ]
        ],
        [
          [
            -1.924
          ]
        ],
        [
          [
            -1.188
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.01 seconds: 
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
    	[ [ 1.8, 1.62 ], [ 0.388, 0.856 ], [ -1.028, -0.084 ], [ -1.844, 0.172 ], [ -0.996, -0.884 ] ],
    	[ [ -0.576, 1.784 ], [ -0.224, -0.836 ], [ -1.748, -1.896 ], [ -0.416, 1.5 ], [ 1.86, -0.16 ] ],
    	[ [ 0.328, -1.2 ], [ 0.596, 0.032 ], [ 1.056, 1.252 ], [ 0.084, 1.832 ], [ 1.828, -0.88 ] ],
    	[ [ 1.388, -1.876 ], [ -1.288, 1.56 ], [ -1.504, 0.768 ], [ 0.752, -0.712 ], [ -1.48, 0.288 ] ],
    	[ [ 0.508, 0.08 ], [ -0.656, -0.368 ], [ 0.92, -1.46 ], [ -1.692, 0.084 ], [ -1.62, -1.744 ] ]
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
Code from [StandardLayerTests.java:92](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L92) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "9d64bdfb-8ede-4599-ab2b-9cedbcf103e7",
      "isFrozen": false,
      "name": "ConvolutionLayer/9d64bdfb-8ede-4599-ab2b-9cedbcf103e7",
      "filter": [
        [
          [
            0.08
          ]
        ],
        [
          [
            1.404
          ]
        ],
        [
          [
            -1.924
          ]
        ],
        [
          [
            -1.188
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
    	[ [ 0.82, 1.492 ], [ 1.244, 1.66 ], [ -1.848, -1.212 ], [ 1.32, 0.756 ], [ -1.676, 1.904 ] ],
    	[ [ 0.14, -1.612 ], [ 1.684, 0.264 ], [ 1.996, 1.228 ], [ 1.48, -0.552 ], [ 1.676, 0.684 ] ],
    	[ [ -0.84, -0.056 ], [ -1.912, -0.188 ], [ 0.476, 1.06 ], [ 1.26, -0.732 ], [ -0.356, -0.368 ] ],
    	[ [ 0.764, -1.248 ], [ -1.972, -0.484 ], [ -1.608, -0.908 ], [ 1.592, -0.632 ], [ 1.732, -1.188 ] ],
    	[ [ 0.616, -0.288 ], [ 1.156, 1.092 ], [ 1.428, 1.144 ], [ 0.536, -1.22 ], [ -0.288, 0.496 ] ]
    ]]
    Subject Output: [
    	[ [ 0.0, 0.0 ], [ 0.0, 0.0 ], [ 0.0
```
...[skipping 1250 bytes](etc/55.txt)...
```
    .39016, 2.201904 ], [ -0.977344, -0.9935999999999999 ] ]
    ]
    Error: [
    	[ [ 2.805008, 0.6212160000000001 ], [ 3.0943199999999997, 0.2255039999999999 ], [ -2.1840479999999998, 1.1547360000000002 ], [ 1.348944, -0.9551520000000001 ], [ 3.797376, 4.615055999999999 ] ],
    	[ [ -3.112688, -2.111616 ], [ 0.373216, -2.0507039999999996 ], [ 2.202992, -1.34352 ], [ -1.1804480000000002, -2.7336959999999997 ], [ 1.181936, -1.5405119999999997 ] ],
    	[ [ -0.040544000000000004, 1.112832 ], [ -0.208752, 2.4611039999999997 ], [ 2.00136, 0.5909760000000001 ], [ -1.5091679999999998, -2.6386559999999997 ], [ -0.6795519999999999, 0.06263999999999996 ] ],
    	[ [ -2.462272, -2.5552799999999998 ], [ -0.7734559999999999, 2.1936959999999996 ], [ -1.618352, 1.1789280000000002 ], [ -1.343328, -2.9859839999999997 ], [ -2.4242719999999998, -3.8430719999999994 ] ],
    	[ [ -0.6033919999999999, -1.2070079999999999 ], [ 2.008528, -0.32572799999999974 ], [ 2.086816, -0.6458399999999997 ], [ -2.39016, -2.201904 ], [ 0.977344, 0.9935999999999999 ] ]
    ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=1.6951e+00 +- 1.0446e+00 [4.0544e-02 - 4.6151e+00] (50#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (50#)}
    	at com.simiacryptus.mindseye.test.EquivalencyTester.test(EquivalencyTester.java:66)
    	at com.simiacryptus.mindseye.test.StandardLayerTests.lambda$test$8(StandardLayerTests.java:94)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:157)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:72)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.test.StandardLay
```
...[skipping 1337 bytes](etc/56.txt)...
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



