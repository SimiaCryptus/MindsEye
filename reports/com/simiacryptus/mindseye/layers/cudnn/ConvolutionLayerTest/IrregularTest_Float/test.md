# ConvolutionLayer
## IrregularTest_Float
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
      "id": "c7c906e4-dc0c-439d-bc8e-2f7122c0f870",
      "isFrozen": false,
      "name": "ConvolutionLayer/c7c906e4-dc0c-439d-bc8e-2f7122c0f870",
      "filter": [
        [
          [
            -1.78,
            -1.848,
            -0.644
          ],
          [
            0.132,
            -0.552,
            -0.576
          ],
          [
            0.948,
            -0.668,
            0.504
          ]
        ],
        [
          [
            1.404,
            0.088,
            -1.696
          ],
          [
            0.152,
            -0.504,
            -0.42
          ],
          [
            -0.888,
            -1.056,
            -0.14
          ]
        ],
        [
          [
            1.612,
            -1.7,
            1.864
          ],
          [
            -0.184,
            1.964,
            0.848
          ],
          [
            -0.372,
            -0.384,
            -1.012
          ]
        ],
        [
          [
            0.904,
            -0.2,
            -1.852
          ],
          [
            -1.848,
            1.036,
            -0.4
          ],
          [
            1.936,
            1.008,
            -0.724
          ]
        ],
        [
       
```
...[skipping 5112 bytes](etc/61.txt)...
```
        [
          [
            -1.352,
            -1.556,
            1.056
          ],
          [
            1.888,
            -1.604,
            -0.04
          ],
          [
            1.324,
            -1.18,
            0.096
          ]
        ],
        [
          [
            -1.544,
            -1.672,
            1.768
          ],
          [
            1.644,
            0.452,
            0.532
          ],
          [
            -0.932,
            0.584,
            1.42
          ]
        ],
        [
          [
            -0.784,
            1.316,
            0.228
          ],
          [
            0.924,
            -1.424,
            -0.068
          ],
          [
            -0.812,
            -1.104,
            0.904
          ]
        ],
        [
          [
            1.384,
            1.176,
            -0.38
          ],
          [
            1.896,
            1.66,
            -1.68
          ],
          [
            1.824,
            1.372,
            -1.908
          ]
        ],
        [
          [
            -0.032,
            -0.64,
            0.976
          ],
          [
            1.376,
            0.564,
            -0.488
          ],
          [
            0.512,
            0.408,
            -1.536
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    	[ [ -0.296, -1.712, -0.788, -1.88, -1.064, -1.304, 1.928 ], [ -1.68, -1.74, 0.736, 0.912, -0.788, -0.048, -1.136 ], [ -0.384, 0.232, -1.928, -0.976, -0.704, -1.108, -1.936 ], [ 1.036, 1.836, 0.196, 0.004, 1.52, -0.64, 0.004 ], [ 0.008, 0.552, 0.952, -0.42, -0.796, 0.456, 0.412 ] ],
    	[ [ 0.772, 1.636, -0.508, 0.584, -1.668, 0.976, 0.32 ], [ -1.056, -0.228, 0.768, 0.608, -1.532, 0.044, -0.648 ], [ -1.564, 1.16, 0.456, -0.028, -0.876, 1.276, -1.132 ], [ 1.06, -1.56, -1.668, 0.528, -1.004, -1.228, -0.136 ], [ -0.344, 0.3, -1.192, 0.04, -1.84, -1.0, -0.152 ] ],
    	[ [ -0.8, -1.792, -1.412, -1.664, 1.912, 0.636, -1.416 ], [ 1.428, -0.588, 1.92, -1.828, 1.796, -1.816, -0.82 ], [ -0.796, -1.432, -0.508, -1.744, -1.872, 1.34, -1.992 ], [ -0.864, -1.276, 0.48, 0.732, 0.268, -0.796, -1.724 ], [ -0.896, -1.872, 0.784, -0.876, -0.044, -1.512, -0.072 ] ],
    	[ [ 0.944, 0.264, -1.18, -1.332, 0.544, -1.504, 1.344 ], [ 1.908, -1.476, -0.336, 0.336, 0.904, 1.828, 0.76 ], [ 0.24, 0.092, 0.796, 1.38,
```
...[skipping 1208 bytes](etc/62.txt)...
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
Code from [StandardLayerTests.java:92](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L92) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "9eb4a831-f567-435d-91d4-31ed11a9540c",
      "isFrozen": false,
      "name": "ConvolutionLayer/9eb4a831-f567-435d-91d4-31ed11a9540c",
      "filter": [
        [
          [
            -1.78,
            -1.848,
            -0.644
          ],
          [
            0.132,
            -0.552,
            -0.576
          ],
          [
            0.948,
            -0.668,
            0.504
          ]
        ],
        [
          [
            1.404,
            0.088,
            -1.696
          ],
          [
            0.152,
            -0.504,
            -0.42
          ],
          [
            -0.888,
            -1.056,
            -0.14
          ]
        ],
        [
          [
            1.612,
            -1.7,
            1.864
          ],
          [
            -0.184,
            1.964,
            0.848
          ],
          [
            -0.372,
            -0.384,
            -1.012
          ]
        ],
        [
          [
            0.904,
            -0.2,
            -1.852
          ],
          [
            -1.848,
            1.036,
            -0.4
          ],
          [
            1.936,
            1.008,
            -0.724
          ]
        ],
        [
     
```
...[skipping 12185 bytes](etc/63.txt)...
```
    96, 6.312656000000001, 15.448127999999999, 0.25412800000000046 ] ],
    	[ [ -5.579967999999999, -18.832976000000002, 16.850464, 9.606239999999998, -9.7584 ], [ -1.8829120000000006, -6.410064000000001, -13.018704, 11.703856, -0.13398400000000132 ], [ -3.691327999999997, 6.308655999999998, 7.5837759999999985, 6.5498400000000006, 23.518255999999994 ], [ -3.175791999999998, 0.9750080000000001, 21.058783999999996, -18.574496000000003, -10.176895999999994 ], [ -19.829056000000005, 3.957007999999999, -6.6672959999999994, -21.445584, -9.08784 ] ],
    	[ [ -8.403344, 7.887599999999999, -2.6693599999999984, 7.1257920000000015, 2.2454400000000003 ], [ 6.684031999999998, -0.6198560000000013, -10.191856, 7.158623999999999, -4.334687999999998 ], [ 3.231120000000001, 8.410864, -3.1656480000000013, 4.869344, 6.5900159999999985 ], [ -3.607472000000003, 4.635087999999999, -14.238911999999994, 11.828367999999994, -8.257728000000002 ], [ 9.84416, -6.216543999999999, -1.5253920000000005, -10.479087999999997, -4.2846079999999995 ] ]
    ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=7.3525e+00 +- 5.5851e+00 [4.0416e-02 - 2.5019e+01] (125#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (125#)}
    	at com.simiacryptus.mindseye.test.EquivalencyTester.test(EquivalencyTester.java:66)
    	at com.simiacryptus.mindseye.test.StandardLayerTests.lambda$test$8(StandardLayerTests.java:94)
    	at com.simiacryptus.util.io.NotebookOutput.lambda$code$1(NotebookOutput.java:157)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$null$1(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:59)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:72)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.test.StandardL
```
...[skipping 1257 bytes](etc/64.txt)...
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



