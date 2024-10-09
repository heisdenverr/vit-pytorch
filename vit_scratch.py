import PIL.Image

class Preprocess:
  
  def __init__(self, image):
    self.image = image

    if isinstance(self.image, PIL.Image.Image):
      print("Image loaded successfully")
    else:

      try:
        
        with PIL.Image.open(f"{self.image}") as img:
          self.image = img
          print("Image loaded successfully!.")
          

      except (FileNotFoundError, IOError) as e:
        print(f"Error loading image: {e}")


  def split_image(self, patch_size=16):
    height, width = self.image.size
    if height != 224 or width != 224:
      self.image = self.image.resize((224, 224))
      print(f"Image resized to: {self.image.size}")

    pw, ph = int(height / patch_size), int(height / patch_size)
    patches = []

    for y in range(0, height, ph):
      for x in range(0, width, pw):

        box = (x, y, x + pw, y + ph)
        patch = self.image.crop(box)

        patches.append(patch)

    return patches


class PatchEmbed:


    """
      A class that handles the embedding of image patches for a Vision Transformer model.
      
      Attributes:
      ----------
      num_patches : int
          The number of image patches (or tokens).
      D : int
          The dimensionality of each token.
      cls_token : numpy.ndarray
          The class token used for classification, initialized randomly.
      pos_embed : numpy.ndarray
          The positional embedding matrix, initialized randomly.
    """
      
    def __init__(self,
               num_patches=196,
               D=768):
    
    
      """
          Initializes the PatchEmbed class with the specified number of patches and token dimensionality.
          
          Parameters:
          ----------
          num_patches : int, optional
              The number of image patches to embed (default is 196).
          D : int, optional
              The dimensionality of the class token and positional embeddings (default is 768).
          """

      self.num_patches = num_patches
      self.D = D

      self.cls_token = np.random.randn(1, 1, self.D)
      
      self.pos_embed = np.random.randn(1, self.num_patches+1, self.D)

      self.cls_ = np.tile(self.cls_token, (1, 1, 1))  # This replicates cls_token to shape (1, 1, D)

      

    def forward(self, x:np.array):
    

      """
          Performs the forward pass through the embedding layer.
          
          Parameters:
          ----------
          x : numpy.ndarray
              Input array representing flattened patches of shape (num_patches, height * width * channels).
          
          Returns:
          -------
          tuple
              The shape of the output after adding the class token and positional embeddings.
          """


      x = x.reshape(self.num_patches, -1)

      x = np.expand_dims(x, axis=0)

      x = np.concatenate((self.cls_, x), axis=1)
      x = x + self.pos_embed

      return x




class TransformerBlock:
    """
    A class representing a transformer block used in Vision Transformers.
    
    Attributes:
    ----------
    D : int
        The dimensionality of the input tokens.
    qw : numpy.ndarray
        The weight matrix for the query transformation.
    kw : numpy.ndarray
        The weight matrix for the key transformation.
    vw : numpy.ndarray
        The weight matrix for the value transformation.
    """
    
    def __init__(self, D=768):
        """
        Initializes the TransformerBlock with the specified token dimensionality.
        
        Parameters:
        ----------
        D : int, optional
            The dimensionality of the input tokens (default is 768).
        """
        self.D = D
        self.qw = np.random.randn(self.D, self.D)  # Weight matrix for queries
        self.kw = np.random.randn(self.D, self.D)  # Weight matrix for keys
        self.vw = np.random.randn(self.D, self.D)  # Weight matrix for values
        
    def get_weights_shape(self):
        """
        Returns the shapes of the weight matrices.
        
        Returns:
        -------
        tuple
            The shapes of the query, key, and value weight matrices.
        """
        return self.qw.shape, self.kw.shape, self.vw.shape


    def qkv_projection(self, x):
      """ 1, 196, 768 --> 768, 768 """
      Q = np.dot(x, self.qw) // 255
      K = np.dot(x, self.kw) // 255
      V = np.dot(x, self.vw) // 255

      return Q, K, V

    def forward(self, x):

      q_, k_, v_ = self.qkv_projection(x)
      
      attn_score = np.dot(q_, np.reshape(k_, (1, self.D, 197)))
      x_s = np.exp(attn_score) / np.sum(np.exp(attn_score))

      x = np.matmul((xs / self.D **0.5), x_s)
      print(x)
      

testing = TransformerBlock()
import torch

with torch.inference_mode():
  y = testing.forward(x=ra)
