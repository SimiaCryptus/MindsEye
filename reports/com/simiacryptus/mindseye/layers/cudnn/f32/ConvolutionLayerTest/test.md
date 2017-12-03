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
      "id": "e2d0bffa-47dc-4875-864f-3d3d00000036",
      "isFrozen": false,
      "name": "ConvolutionLayer/e2d0bffa-47dc-4875-864f-3d3d00000036",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          -0.768,
          0.62,
          -0.848,
          -1.304
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
    	[ [ 0.184, 0.44 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.5144320130348206, -0.4596799910068512 ] ]
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
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000003d",
      "isFrozen": false,
      "name": "ConvolutionLayer/e2d0bffa-47dc-4875-864f-3d3d0000003d",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          -0.768,
          0.62,
          -0.848,
          -1.304
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
    	[ [ 0.184, 0.44 ] ]
    ]
    Error: [
    	[ [ -1.3034820556256932E-8, 8.993148836733411E-9 ] ]
    ]
    Accuracy:
    absoluteTol: 1.1014e-08 +- 2.0208e-09 [8.9931e-09 - 1.3035e-08] (2#)
    relativeTol: 1.1226e-08 +- 1.4436e-09 [9.7820e-09 - 1.2669e-08] (2#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.03 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [
    	[ [ 0.184, 0.44 ] ]
    ]
    Output: [
    	[ [ -0.5144320130348206, -0.4596799910068512 ] ]
    ]
    Measured: [ [ -0.7677078247070312, 0.6198883056640625 ], [ -0.8481740951538086, -1.3044476509094238 ] ]
    Implemented: [ [ -0.7680000066757202, 0.6200000047683716 ], [ -0.8479999899864197, -1.3040000200271606 ] ]
    Error: [ [ 2.9218196868896484E-4, -1.1169910430908203E-4 ], [ -1.7410516738891602E-4, -4.476308822631836E-4 ] ]
    Learning Gradient for weight set 0
    Inputs: [
    	[ [ 0.184, 0.44 ] ]
    ]
    Outputs: [
    	[ [ -0.5144320130348206, -0.4596799910068512 ] ]
    ]
    Measured Gradient: [ [ 0.18417835235595703, 0.0 ], [ 0.0, 0.18388032913208008 ], [ 0.4404783248901367, 0.0 ], [ 0.0, 0.4398822784423828 ] ]
    Implemented Gradient: [ [ 0.18400000035762787, 0.0 ], [ 0.0, 0.18400000035762787 ], [ 0.4399999976158142, 0.0 ], [ 0.0, 0.4399999976158142 ] ]
    Error: [ [ 1.783519983291626E-4, 0.0 ], [ 0.0, -1.1967122554779053E-4 ], [ 4.7832727432250977E-4, 0.0 ], [ 0.0, -1.1771917343139648E-4 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.5997e-04 +- 1.6078e-04 [0.0000e+00 - 4.7833e-04] (12#)
    relativeTol: 2.5517e-04 +- 1.6466e-04 [9.0088e-05 - 5.4326e-04] (8#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.29 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 10.9528 +- 1.3072 [9.3758 - 15.0326]
    Learning performance: 6.9766 +- 1.1362 [5.8563 - 14.9243]
    
```

