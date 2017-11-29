# SimpleConvolutionLayer
## MultiBand
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-8909000002fa",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-8909000002fa",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          -1.876,
          -1.052,
          -1.0,
          -0.68,
          -0.644,
          -0.46,
          -0.512,
          -1.448,
          1.516
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    	[ [ -0.036, 1.368, 1.388 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -2.7596001625061035, -1.4949920177459717, 0.14177604019641876 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-8909000002fb",
      "isFrozen": false,
      "name": "ConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-8909000002fb",
      "filter": {
        "dimensions": [
          1,
          1,
          9
        ],
        "data": [
          -1.876,
          -0.68,
          -0.512,
          -1.052,
          -0.644,
          -1.448,
          -1.0,
          -0.46,
          1.516
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
    	[ [ -0.036, 1.368, 1.388 ] ]
    ]
    Error: [
    	[ [ -1.6250610368473417E-7, -1.77459715811068E-8, 4.0196419082150214E-8 ] ]
    ]
    Accuracy:
    absoluteTol: 7.3483e-08 +- 6.3613e-08 [1.7746e-08 - 1.6251e-07] (3#)
    relativeTol: 5.9046e-08 +- 5.9270e-08 [5.9351e-09 - 1.4176e-07] (3#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.02 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: SimpleConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-8909000002fa
    Inputs: [
    	[ [ -0.036, 1.368, 1.388 ] ]
    ]
    output=[
    	[ [ -2.7596001625061035, -1.4949920177459717, 0.14177604019641876 ] ]
    ]
    measured/actual: [ [ -1.8739700317382812, -0.6806850433349609, -0.5114078521728516 ], [ -1.0514259338378906, -0.6449222564697266, -1.4472007751464844 ], [ -1.0013580322265625, -0.4601478576660156, 1.516193151473999 ] ]
    implemented/expected: [ [ -1.8760000467300415, -0.6800000071525574, -0.5120000243186951 ], [ -1.0520000457763672, -0.6439999938011169, -1.4479999542236328 ], [ -1.0, -0.46000000834465027, 1.5160000324249268 ] ]
    error: [ [ 0.002030014991760254, -6.850361824035645E-4, 5.921721458435059E-4 ], [ 5.741119384765625E-4, -9.222626686096191E-4, 7.991790771484375E-4 ], [ -0.0013580322265625, -1.4784932136535645E-4, 1.9311904907226562E-4 ] ]
    Component: SimpleConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-8909000002fa
    Inputs: [
    	[ [ -0.036, 1.368, 1.388 ] ]
    ]
    Outputs: [
    	[ [ -2.7596001625061035, -1.4949920177459717, 0.14177604019641876 ] ]
    ]
    Measured Gradient: [ [ -0.035762786865234375, 0.0, 0.0 ], [ 1.3685226440429688, 0.0, 0.0 ], [ 1.3875961303710938, 0.0, 0.0 ], [ 0.0, -0.035762786865234375, 0.0 ], [ 0.0, 1.367330551147461, 0.0 ], [ 0.0, 1.3875961303710938, 0.0 ], [ 0.0, 0.0, -0.035762786865234375 ], [ 0.0, 0.0, 1.367330551147461 ], [ 0.0, 0.0, 1.3881921768188477 ] ]
    Implemented Gradient: [ [ -0.035999998450279236, 0.0, 0.0 ], [ 1.3680000305175781, 0.0, 0.0 ], [ 1.3880000114440918, 0.0, 0.0 ], [ 0.0, -0.035999998450279236, 0.0 ], [ 0.0, 1.3680000305175781, 0.0 ], [ 0.0, 1.3880000114440918, 0.0 ], [ 0.0, 0.0, -0.035999998450279236 ], [ 0.0, 0.0, 1.3680000305175781 ], [ 0.0, 0.0, 1.3880000114440918 ] ]
    Error: [ [ 2.3721158504486084E-4, 0.0, 0.0 ], [ 5.22613525390625E-4, 0.0, 0.0 ], [ -4.038810729980469E-4, 0.0, 0.0 ], [ 0.0, 2.3721158504486084E-4, 0.0 ], [ 0.0, -6.694793701171875E-4, 0.0 ], [ 0.0, -4.038810729980469E-4, 0.0 ], [ 0.0, 0.0, 2.3721158504486084E-4 ], [ 0.0, 0.0, -6.694793701171875E-4 ], [ 0.0, 0.0, 1.9216537475585938E-4 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.0208e-04 +- 4.4346e-04 [0.0000e+00 - 2.0300e-03] (36#)
    relativeTol: 8.1934e-04 +- 1.1287e-03 [6.3690e-05 - 3.3055e-03] (18#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.13 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.8758 +- 0.7464 [3.1861 - 7.3097]
    Learning performance: 3.9203 +- 1.2324 [2.8612 - 12.1543]
    
```

