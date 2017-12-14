# SimpleConvolutionLayer
## Image_Float
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.SimpleConvolutionLayer",
      "id": "61a5c877-3874-40e2-a96b-4aabb173009d",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/61a5c877-3874-40e2-a96b-4aabb173009d",
      "filter": [
        [
          [
            -0.496,
            -0.94,
            0.564
          ],
          [
            -1.108,
            0.176,
            -1.12
          ],
          [
            -1.528,
            0.404,
            -1.412
          ]
        ],
        [
          [
            0.94,
            1.644,
            -1.772
          ],
          [
            -1.128,
            -1.032,
            0.58
          ],
          [
            -1.528,
            -0.828,
            -1.772
          ]
        ],
        [
          [
            -0.72,
            -1.516,
            -0.26
          ],
          [
            -0.732,
            -0.944,
            0.408
          ],
          [
            1.716,
            -0.856,
            -0.74
          ]
        ],
        [
          [
            0.02,
            0.948,
            1.432
          ],
          [
            1.252,
            -1.492,
            -0.464
          ],
          [
            0.592,
            -0.88,
            0.74
          ]
        ]
```
...[skipping 19 bytes](etc/86.txt)...
```
         -0.384,
            -1.392,
            1.024
          ],
          [
            1.808,
            1.36,
            -1.448
          ],
          [
            1.06,
            1.208,
            0.248
          ]
        ],
        [
          [
            -0.672,
            -0.272,
            0.656
          ],
          [
            0.5,
            1.568,
            -0.176
          ],
          [
            -1.816,
            -1.588,
            -0.356
          ]
        ],
        [
          [
            0.944,
            -0.26,
            -1.852
          ],
          [
            1.54,
            0.404,
            -1.832
          ],
          [
            -1.024,
            -1.868,
            1.036
          ]
        ],
        [
          [
            -0.276,
            1.396,
            0.18
          ],
          [
            -0.764,
            0.056,
            -1.388
          ],
          [
            0.392,
            1.064,
            -0.792
          ]
        ],
        [
          [
            0.14,
            1.56,
            -0.24
          ],
          [
            1.684,
            -1.244,
            -1.116
          ],
          [
            1.66,
            1.304,
            -0.588
          ]
        ]
      ],
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    	[ [ -1.408, -0.228, 1.452 ], [ 0.588, 1.616, 0.476 ], [ 0.932, 0.484, 1.792 ] ],
    	[ [ -1.308, -1.232, 1.984 ], [ 0.108, 1.432, 0.928 ], [ -1.228, -1.364, 0.54 ] ],
    	[ [ 1.232, 0.428, -0.832 ], [ 0.332, -0.44, -1.144 ], [ 0.372, 1.784, -1.252 ] ]
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
Code from [StandardLayerTests.java:92](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L92) executed in 0.00 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "47e7d9ca-6f34-450b-b465-3d69e3d0693f",
      "isFrozen": false,
      "name": "ConvolutionLayer/47e7d9ca-6f34-450b-b465-3d69e3d0693f",
      "filter": [
        [
          [
            -0.496,
            -0.94,
            0.564
          ],
          [
            -1.108,
            0.176,
            -1.12
          ],
          [
            -1.528,
            0.404,
            -1.412
          ]
        ],
        [
          [
            0.02,
            0.948,
            1.432
          ],
          [
            1.252,
            -1.492,
            -0.464
          ],
          [
            0.592,
            -0.88,
            0.74
          ]
        ],
        [
          [
            0.944,
            -0.26,
            -1.852
          ],
          [
            1.54,
            0.404,
            -1.832
          ],
          [
            -1.024,
            -1.868,
            1.036
          ]
        ],
        [
          [
            0.94,
            1.644,
            -1.772
          ],
          [
            -1.128,
            -1.032,
            0.58
          ],
          [
            -1.528,
            -0.828,
            -1.772
          ]
        ],
        [
       
```
...[skipping 1460 bytes](etc/87.txt)...
```
    0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ]
    ]
    Reference Output: [
    	[ [ 7.637487999999999, -0.5958400000000003, -7.221184 ], [ -0.4134399999999997, 3.3377280000000007, -9.147616 ], [ 1.2112800000000004, 1.3404480000000008, -3.3076159999999994 ] ],
    	[ [ 4.832832, -3.6918720000000005, 6.4054079999999995 ], [ 6.334831999999999, 3.340256, 0.727424 ], [ -3.001472, 3.272480000000001, 6.393295999999999 ] ],
    	[ [ -9.814128, 5.586864, 4.946512 ], [ -0.5749759999999997, -4.0918719999999995, 11.27496 ], [ -2.557296, 1.313296, -5.5996 ] ]
    ]
    Error: [
    	[ [ -7.637487999999999, 0.5958400000000003, 7.221184 ], [ 0.4134399999999997, -3.3377280000000007, 9.147616 ], [ -1.2112800000000004, -1.3404480000000008, 3.3076159999999994 ] ],
    	[ [ -4.832832, 3.6918720000000005, -6.4054079999999995 ], [ -6.334831999999999, -3.340256, -0.727424 ], [ 3.001472, -3.272480000000001, -6.393295999999999 ] ],
    	[ [ 9.814128, -5.586864, -4.946512 ], [ 0.5749759999999997, 4.0918719999999995, -11.27496 ], [ 2.557296, -1.313296, 5.5996 ] ]
    ]
    
```

Returns: 

```
    java.lang.AssertionError: ToleranceStatistics{absoluteTol=4.3693e+00 +- 2.9238e+00 [4.1344e-01 - 1.1275e+01] (27#), relativeTol=1.0000e+00 +- 0.0000e+00 [1.0000e+00 - 1.0000e+00] (27#)}
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
...[skipping 1255 bytes](etc/88.txt)...
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



