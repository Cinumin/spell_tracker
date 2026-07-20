'''
create an openGL window in the size of the camera feed, I specify vertices for all the corners and create 2 indices for the two triangles that make up the entire screen. I load the texture as done in the github link above with glTexImage2D(but with image instead of buf), define a simple texture shader and then render it using a uniform value for the texture.

Tricky stuff I ran into:
OpenCV pads images however it wants, add the line glPixelStorei(GL_UNPACK_ALIGNMENT, (image.step & 3) ? 1 : 4);
before you do any openGL-functions which often require constant defined steps.

Flip the image before you do anything. OpenGL sees the y-axis as going up while OpenCV sees it as going down.

flip(image, image, 0);
'''
// Copyright (c) 2021 F. Lotfi & D. Kane All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "texture.h"

GLuint Texture::genTexture(PixelBuffer &buf) {
  if (!mVisited) {
    this->mTexture = matToTexture(buf, GL_NEAREST, GL_NEAREST, GL_CLAMP);
    mVisited = true;
  } else {
    glDeleteTextures(1, &this->mTexture);
    this->mTexture = matToTexture(buf, GL_NEAREST, GL_NEAREST, GL_CLAMP);
  }
  return this->mTexture;
}

GLuint Texture::matToTexture(PixelBuffer &buf, GLenum minFilter,
                             GLenum magFilter, GLenum wrapFilter) {
  // Generate a number for our textureID's unique handle
  // static int visited = 0;
  GLuint textureID;
  glGenTextures(1, &textureID);

  // Bind to our texture handle
  glBindTexture(GL_TEXTURE_2D, textureID);

  // Catch silly-mistake texture interpolation method for magnification
  if (magFilter == GL_LINEAR_MIPMAP_LINEAR ||
      magFilter == GL_LINEAR_MIPMAP_NEAREST ||
      magFilter == GL_NEAREST_MIPMAP_LINEAR ||
      magFilter == GL_NEAREST_MIPMAP_NEAREST) {
    std::cout << "You can't use MIPMAPs for magnification - setting filter to "
                 "GL_LINEAR"
              << std::endl;
    magFilter = GL_LINEAR;
  }

  // Set texture interpolation methods for minification and magnification
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

  // Set texture clamping method
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

  // Set incoming texture format to:
  // Work out other mappings as required ( there's a list in comments in main()
  // )
  GLenum inputColourFormat = GL_RGB;
  switch (buf.channels()) {
  case ColorChannels::BW:
    inputColourFormat = GL_LUMINANCE;
    break;
  case ColorChannels::RGB:
    inputColourFormat = GL_RGB;
    break;
  case ColorChannels::RGBA:
    inputColourFormat = GL_RGBA;
    break;
  }

  GLenum datatype = GL_UNSIGNED_BYTE;
  switch (buf.depth()) {
  case DataDepth::PB_32F:
    datatype = GL_FLOAT;
    break;
  case DataDepth::PB_64F:
    datatype = GL_DOUBLE;
    break;
  case DataDepth::PB_32S:
    datatype = GL_INT;
    break;
  case DataDepth::PB_32U:
    datatype = GL_UNSIGNED_INT;
    break;
  case DataDepth::PB_16U:
    datatype = GL_UNSIGNED_SHORT;
    break;
  case DataDepth::PB_16S:
    datatype = GL_SHORT;
    break;
  case DataDepth::PB_8U:
    datatype = GL_UNSIGNED_BYTE;
    break;
  case DataDepth::PB_8S:
    datatype = GL_BYTE;
    break;
  }

  // Create the texture
  // if(!visited)
  glTexImage2D(GL_TEXTURE_2D, // Type of texture
               0,      // Pyramid level (for mip-mapping) - 0 is the top level
               GL_RGB, // Internal colour format to convert to
               buf.width(),  // Image width  i.e. 640
               buf.height(), // Image height i.e. 480
               0,            // Border width in pixels (can either be 1 or 0)
               inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA,
                                  // GL_BGR etc.)
               datatype,          // Image data type
               buf.data());       // The actual image data itself

  // If we're using mipmaps then generate them. Note: This requires OpenGL 3.0
  // or higher
  if (minFilter == GL_LINEAR_MIPMAP_LINEAR ||
      minFilter == GL_LINEAR_MIPMAP_NEAREST ||
      minFilter == GL_NEAREST_MIPMAP_LINEAR ||
      minFilter == GL_NEAREST_MIPMAP_NEAREST) {
    glGenerateMipmap(GL_TEXTURE_2D);
  }

  return textureID;
}
