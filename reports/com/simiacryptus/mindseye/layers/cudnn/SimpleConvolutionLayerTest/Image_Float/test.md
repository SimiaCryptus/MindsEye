# SimpleConvolutionLayer
## Image_Float
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer",
      "id": "3bce09d4-ed5e-404d-8004-92e7bc43c73e",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/3bce09d4-ed5e-404d-8004-92e7bc43c73e",
      "filter": [
        [
          [
            -1.232,
            0.592,
            -1.212
          ],
          [
            -0.192,
            -1.076,
            1.516
          ],
          [
            -0.76,
            -0.712,
            1.332
          ]
        ],
        [
          [
            -1.884,
            0.04,
            1.408
          ],
          [
            0.308,
            0.82,
            0.4
          ],
          [
            -1.076,
            1.632,
            -0.044
          ]
        ],
        [
          [
            -1.984,
            1.58,
            0.156
          ],
          [
            -1.872,
            -1.64,
            0.188
          ],
          [
            0.108,
            -1.872,
            1.528
          ]
        ],
        [
          [
            -0.64,
            0.856,
            1.192
          ],
          [
            0.3,
            -1.92,
            -0.384
          ],
          [
            -0.324,
            0.524,
            -0.272
          ]
        ],
       
```
...[skipping 14 bytes](etc/48.txt)...
```
         0.232,
            -1.588,
            1.688
          ],
          [
            -0.656,
            1.492,
            1.284
          ],
          [
            -1.024,
            1.148,
            -0.912
          ]
        ],
        [
          [
            0.836,
            1.108,
            -1.852
          ],
          [
            -1.66,
            0.724,
            -0.128
          ],
          [
            1.452,
            0.216,
            -0.628
          ]
        ],
        [
          [
            1.736,
            -1.008,
            -0.108
          ],
          [
            1.78,
            0.092,
            0.992
          ],
          [
            -1.548,
            -1.4,
            0.552
          ]
        ],
        [
          [
            -0.932,
            1.16,
            -1.1
          ],
          [
            1.436,
            1.232,
            0.256
          ],
          [
            -1.432,
            -0.972,
            -0.028
          ]
        ],
        [
          [
            -0.26,
            1.28,
            1.612
          ],
          [
            -0.448,
            -0.104,
            0.772
          ],
          [
            -1.4,
            1.296,
            -1.088
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    	[ [ 1.644, 0.904, 1.272 ], [ -1.96, -0.328, 0.632 ], [ 1.228, -0.892, 0.044 ] ],
    	[ [ -1.552, 0.588, -0.084 ], [ 0.88, 1.264, 0.2 ], [ -1.752, 0.24, -1.404 ] ],
    	[ [ 0.608, 1.54, 1.56 ], [ -0.104, 0.476, -0.544 ], [ -1.092, -0.148, -1.38 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ],
    	[ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
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
      "id": "497ca662-cb10-4ecc-be90-c89c1a87ea94",
      "isFrozen": false,
      "name": "ConvolutionLayer/497ca662-cb10-4ecc-be90-c89c1a87ea94",
      "filter": [
        [
          [
            -1.232,
            0.592,
            -1.212
          ],
          [
            -0.192,
            -1.076,
            1.516
          ],
          [
            -0.76,
            -0.712,
            1.332
          ]
        ],
        [
          [
            -0.64,
            0.856,
            1.192
          ],
          [
            0.3,
            -1.92,
            -0.384
          ],
          [
            -0.324,
            0.524,
            -0.272
          ]
        ],
        [
          [
            1.736,
            -1.008,
            -0.108
          ],
          [
            1.78,
            0.092,
            0.992
          ],
          [
            -1.548,
            -1.4,
            0.552
          ]
        ],
        [
          [
            -1.884,
            0.04,
            1.408
          ],
          [
            0.308,
            0.82,
            0.4
          ],
          [
            -1.076,
            1.632,
            -0.044
          ]
        ],
        [
          [
```
...[skipping 1469 bytes](etc/49.txt)...
```
    0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    ]
    Reference Output: [
    	[ [ -10.176928, -5.282384, -1.646944 ], [ -6.092143999999999, -0.5336319999999991, -6.163647999999999 ], [ 6.52216, 3.1574399999999994, -0.10966399999999951 ] ],
    	[ [ -4.95672, -2.9472480000000005, 10.2972 ], [ -7.171088, 9.553136000000002, -2.969487999999999 ], [ -6.107792, -6.635184000000002, -9.339407999999999 ] ],
    	[ [ 5.158176, -3.639072, 6.101775999999999 ], [ -3.8859839999999983, 9.924752, -2.8731999999999998 ], [ -1.097504, -0.48736000000000035, -6.944000000000001 ] ]
    ]
    Error: [
    	[ [ 10.176928, 5.282384, 1.646944 ], [ 6.092143999999999, 0.5336319999999991, 6.163647999999999 ], [ -6.52216, -3.1574399999999994, 0.10966399999999951 ] ],
    	[ [ 4.95672, 2.9472480000000005, -10.2972 ], [ 7.171088, -9.553136000000002, 2.969487999999999 ], [ 6.107792, 6.635184000000002, 9.339407999999999 ] ],
    	[ [ -5.158176, 3.639072, -6.101775999999999 ], [ 3.8859839999999983, -9.924752, 2.8731999999999998 ], [ 1.097504, 0.48736000000000035, 6.944000000000001 ] ]
    ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=5.1768e+00 +- 3.0213e+00 [1.0966e-01 - 1.0297e+01] (27#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (27#)}
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
...[skipping 1254 bytes](etc/50.txt)...
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



