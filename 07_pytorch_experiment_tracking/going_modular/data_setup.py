
"""
Berisi fungsi untuk mengcreate PyTorch DataLoader untuk data klasifikasi gambar 
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 0

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int = 0
):
  """Buat proses latih dan testing DataLoaders

  Mengambil training dan test direkotri dan mengubahnya menjadi PyTorch datasets dan menjadi PyTorch DataLoaders

  Args:
    train_dir: Path ke direktori data training
    test_dir: Path ke direktori data testing
    transform: torchvision tranform untuk melakukan data training dan testing
    batch_size: Number samples per batch dari DataLoaders.
    num_workers: Angka integer dari jumlah workers per DataLoaders

  Returns:
    Tuple dari (train_dataloader, test_dataloader, class_names).
    Dimana class_names adalah list dari kelas target

    Contoh pemakaian:
      train_dataloader, test_datalaoder, class_names = create_dataloaders
        (train_dir=patah/to/train_dir,
        test_dir=path/to/test_dir,
        transform=your_transform,
        batch_size=32,
        num_worker=0)
  """

  # Gunakan ImageFolder untuk membuat dataset
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Ambil nama kelas
  class_names = train_data.classes

  # Ubah gambar kita ke DataLoaders
  train_dataloader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
  
  test_dataloader = DataLoader(test_data,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers,
                               pin_memory=True)

  return train_dataloader, test_dataloader, class_names

