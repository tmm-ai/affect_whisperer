import Foundation
import AVFoundation

# Find the front-facing camera
devices = AVCaptureDevice.devices()
front_camera = next((d for d in devices if d.position() == AVCaptureDevicePosition.Front), None)

# Configure the camera capture session
session = AVCaptureSession.alloc().init()
input = AVCaptureDeviceInput.deviceInputWithDevice_error_(front_camera, None)
output = AVCaptureStillImageOutput.alloc().init()
session.addInput_(input)
session.addOutput_(output)
session.startRunning()

# Capture an image and save it to a file
connection = output.connectionWithMediaType_(AVMediaTypeVideo)
output.captureStillImageAsynchronouslyFromConnection_completionHandler_(
    connection,
    lambda buffer, error: Foundation.NSData.dataWithData_(AVCaptureStillImageOutput.jpegStillImageNSDataRepresentation_(buffer)).writeToFile_atomically_("photo_mac_test.jpg", True))

# Stop the camera capture session
session.stopRunning()
