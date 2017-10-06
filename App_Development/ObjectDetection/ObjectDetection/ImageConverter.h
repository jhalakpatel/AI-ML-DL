//
//  ImageConverter.h
//  ObjectDetection
//
//  Created by Jhalak Patel on 10/4/17.
//  Copyright © 2017 Jhalak Patel. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

@interface ImageConverter : NSObject

+ (CVPixelBufferRef) pixelBufferFromImage: (CGImageRef) image;

@end
