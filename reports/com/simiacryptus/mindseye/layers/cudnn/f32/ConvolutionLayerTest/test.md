# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "id": "0910987d-3688-428c-a892-e2c400000019",
      "isFrozen": false,
      "name": "ConvolutionLayer/0910987d-3688-428c-a892-e2c400000019",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          0.04,
          -0.108,
          -1.508,
          -0.948,
          1.56,
          1.844,
          -0.352,
          1.616,
          -0.256,
          -0.004,
          -1.02,
          -0.772,
          -1.348,
          0.752,
          1.332,
          1.588,
          -1.068,
          1.972,
          1.336,
          -0.1,
          -0.336,
          -1.7,
          -1.816,
          1.34,
          -0.552,
          1.176,
          -1.412,
          -0.216,
          -0.132,
          -0.832,
          1.86,
          1.56,
          0.944,
          -0.904,
          -1.428,
          0.296
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.01 seconds: 
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
    	[ [ -0.724, 1.86 ], [ -1.46, 0.044 ], [ -1.324, -1.648 ] ],
    	[ [ 0.292, -1.08 ], [ -1.932, 0.616 ], [ -1.72, 1.816 ] ],
    	[ [ -0.468, 0.472 ], [ -0.256, -1.412 ], [ 1.38, 0.312 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -2.0490400791168213, 1.3128001689910889 ], [ 2.6027519702911377, 3.4606239795684814 ], [ -2.496896505355835, 0.0013594627380371094 ] ],
    	[ [ 3.652688503265381, 4.120896339416504 ], [ -4.431248664855957, -1.6579352617263794 ], [ -13.68600082397461, -3.872128486633301 ] ],
    	[ [ 0.3793438673019409, 1.1803040504455566 ], [ 2.4800004959106445, -5.937055587768555 ], [ -1.6014398336410522, -0.3900797367095947 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "0910987d-3688-428c-a892-e2c400000021",
      "isFrozen": false,
      "name": "ConvolutionLayer/0910987d-3688-428c-a892-e2c400000021",
      "filter": {
        "dimensions": [
          3,
          3,
          4
        ],
        "data": [
          0.04,
          -0.108,
          -1.508,
          -0.948,
          1.56,
          1.844,
          -0.352,
          1.616,
          -0.256,
          -0.004,
          -1.02,
          -0.772,
          -1.348,
          0.752,
          1.332,
          1.588,
          -1.068,
          1.972,
          1.336,
          -0.1,
          -0.336,
          -1.7,
          -1.816,
          1.34,
          -0.552,
          1.176,
          -1.412,
          -0.216,
          -0.132,
          -0.832,
          1.86,
          1.56,
          0.944,
          -0.904,
          -1.428,
          0.296
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
    Inputs: [
    	[ [ -0.724, 1.86 ], [ -1.46, 0.044 ], [ -1.324, -1.648 ] ],
    	[ [ 0.292, -1.08 ], [ -1.932, 0.616 ], [ -1.72, 1.816 ] ],
    	[ [ -0.468, 0.472 ], [ -0.256, -1.412 ], [ 1.38, 0.312 ] ]
    ]
    Error: [
    	[ [ -7.91168210945159E-8, 1.6899108867818313E-7 ], [ -2.9708862481214737E-8, -2.0431518699126627E-8 ], [ -5.053558340684106E-7, -5.372619634928886E-7 ] ],
    	[ [ 5.032653804803999E-7, 3.394165046799458E-7 ], [ -6.648559560673561E-7, 7.382736200156614E-7 ], [ -8.239746112082003E-7, -4.866333020991931E-7 ] ],
    	[ [ -1.3269805837490267E-7, 5.044555662081507E-8 ], [ 4.95910645437192E-7, 4.1223144542357204E-7 ], [ 1.6635894728445066E-7, 2.6329040542227844E-7 ] ]
    ]
    Accuracy:
    absoluteTol: 3.5657e-07 +- 2.4675e-07 [2.0432e-08 - 8.2397e-07] (18#)
    relativeTol: 1.1054e-05 +- 4.5235e-05 [2.9520e-09 - 1.9756e-04] (18#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.17 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.3649e-04 +- 2.4338e-04 [0.0000e+00 - 1.9810e-03] (972#)
    relativeTol: 1.6458e-01 +- 3.7026e-01 [3.6884e-07 - 1.0000e+00] (469#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.31 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 11.0327 +- 3.0129 [7.1558 - 19.3044]
    Learning performance: 8.9501 +- 3.0511 [5.5343 - 21.0200]
    
```

