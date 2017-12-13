# ConvolutionLayer
## IrregularTest_Float
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
      "id": "895fa2d2-920e-4898-96e8-f0cd60ae1f46",
      "isFrozen": false,
      "name": "ConvolutionLayer/895fa2d2-920e-4898-96e8-f0cd60ae1f46",
      "filter": [
        [
          [
            -1.396,
            -0.716,
            1.644
          ],
          [
            -0.204,
            -1.644,
            0.44
          ],
          [
            1.78,
            0.052,
            0.22
          ]
        ],
        [
          [
            -0.76,
            0.74,
            -1.976
          ],
          [
            1.004,
            1.648,
            -1.048
          ],
          [
            -0.976,
            -1.116,
            -0.244
          ]
        ],
        [
          [
            0.488,
            0.82,
            0.036
          ],
          [
            -0.276,
            -1.532,
            -1.716
          ],
          [
            -0.808,
            -0.696,
            -1.352
          ]
        ],
        [
          [
            -1.168,
            -0.352,
            0.304
          ],
          [
            1.1,
            -0.42,
            -1.54
          ],
          [
            -1.276,
            1.108,
            0.208
          ]
        ],
        [
         
```
...[skipping 5134 bytes](etc/25.txt)...
```
      [
            -1.896,
            1.596,
            -0.028
          ],
          [
            -1.216,
            0.212,
            0.068
          ],
          [
            1.628,
            -0.368,
            -0.672
          ]
        ],
        [
          [
            0.632,
            -1.924,
            -0.396
          ],
          [
            -1.616,
            -1.804,
            1.848
          ],
          [
            -0.476,
            0.02,
            -0.828
          ]
        ],
        [
          [
            -0.764,
            0.112,
            1.148
          ],
          [
            1.572,
            -0.352,
            0.208
          ],
          [
            1.052,
            -0.296,
            -1.408
          ]
        ],
        [
          [
            -1.436,
            -0.408,
            -1.408
          ],
          [
            0.008,
            0.468,
            -0.436
          ],
          [
            1.792,
            -1.064,
            0.388
          ]
        ],
        [
          [
            1.732,
            1.044,
            0.244
          ],
          [
            -0.364,
            -1.628,
            -0.04
          ],
          [
            -0.848,
            -1.908,
            1.764
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
    	[ [ 0.192, 1.516, 1.284, -0.136, -0.988, -1.616, -1.972 ], [ -1.792, 1.5, 0.12, -1.724, 0.22, -1.556, -0.752 ], [ 1.06, -1.648, -1.984, 1.292, 1.796, 1.452, 1.212 ], [ -1.74, 0.084, 1.3, -1.96, -1.164, 0.188, -0.452 ], [ -0.252, 1.284, -0.188, -0.896, 1.42, 0.392, -0.136 ] ],
    	[ [ -1.048, -1.664, -1.708, 1.228, -1.4, 1.716, 1.804 ], [ -0.572, -0.82, -0.372, 1.148, -1.312, -1.696, -0.524 ], [ 1.364, 1.504, -0.608, -1.832, -1.428, 0.368, 0.812 ], [ -1.432, 0.14, -0.836, -0.228, -0.084, -1.52, -0.832 ], [ 0.116, -1.524, -0.192, -1.556, -1.56, -1.204, 1.384 ] ],
    	[ [ 1.548, 1.992, -0.96, -0.212, 0.368, -0.756, 0.852 ], [ -0.208, 1.424, -0.98, 0.408, 1.988, 0.012, -1.444 ], [ -0.608, 1.236, 0.364, 1.608, 1.172, 1.204, 0.064 ], [ 1.836, -0.592, -1.82, -1.36, -1.864, 1.132, -0.768 ], [ -1.652, 0.88, -1.76, -0.008, 1.66, 0.84, -1.452 ] ],
    	[ [ 1.692, 0.42, -1.48, 1.22, 0.524, -0.524, 0.888 ], [ -0.748, -1.508, 1.964, -1.784, 0.392, -0.988, -1.492 ], [ -1.16, -0.276, 0.604, 0.772, 1.35
```
...[skipping 1205 bytes](etc/26.txt)...
```
    --------
    Derivative: 
    [
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ]
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
      "id": "ddfaf2f3-0707-443a-a094-cd8650f4e441",
      "isFrozen": false,
      "name": "ConvolutionLayer/ddfaf2f3-0707-443a-a094-cd8650f4e441",
      "filter": [
        [
          [
            -1.396,
            -0.716,
            1.644
          ],
          [
            -0.204,
            -1.644,
            0.44
          ],
          [
            1.78,
            0.052,
            0.22
          ]
        ],
        [
          [
            -0.76,
            0.74,
            -1.976
          ],
          [
            1.004,
            1.648,
            -1.048
          ],
          [
            -0.976,
            -1.116,
            -0.244
          ]
        ],
        [
          [
            0.488,
            0.82,
            0.036
          ],
          [
            -0.276,
            -1.532,
            -1.716
          ],
          [
            -0.808,
            -0.696,
            -1.352
          ]
        ],
        [
          [
            -1.168,
            -0.352,
            0.304
          ],
          [
            1.1,
            -0.42,
            -1.54
          ],
          [
            -1.276,
            1.108,
            0.208
          ]
        ],
        [
       
```
...[skipping 12230 bytes](etc/27.txt)...
```
    994, 0.8243200000000006, 5.277120000000002, -18.23632 ] ],
    	[ [ 1.5847359999999986, -14.283664000000003, -1.1685599999999978, 9.993520000000002, 1.4124639999999986 ], [ 9.108544, -2.338592000000001, 3.189408, -5.089199999999999, 21.219952000000006 ], [ 4.604831999999999, -2.809455999999999, 9.427280000000003, -14.022960000000003, -2.3914560000000016 ], [ 8.364624000000001, -6.568128000000001, 3.7221280000000005, -6.505200000000003, 6.3149760000000015 ], [ 15.830943999999997, 1.2221920000000002, 13.337695999999998, -9.188352, -1.1194559999999991 ] ],
    	[ [ -0.32108799999999965, -4.901503999999998, 6.469728, 7.087599999999999, 2.277712 ], [ 3.8546880000000017, -5.7150560000000015, 13.070112, 6.481856000000003, 7.289295999999998 ], [ -10.839968, 4.072943999999999, 11.386479999999999, -6.001296, -17.059359999999998 ], [ -3.4366720000000006, 13.104175999999994, -1.3633760000000006, -2.1852800000000023, 8.092911999999998 ], [ 2.298512, -4.077712, -0.49883199999999983, -2.9115039999999994, -1.2166560000000006 ] ]
    ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=7.0210e+00 +- 5.5352e+00 [3.2109e-01 - 3.3359e+01] (125#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (125#)}
    	at com.simiacryptus.mindseye.test.EquivalencyTester.test(EquivalencyTester.java:66)
    	at com.simiacryptus.mindseye.test.StandardLayerTests.lambda$test$8(StandardLayerTests.java:95)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:157)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:82)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.test.StandardL
```
...[skipping 1256 bytes](etc/28.txt)...
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



