Lightweight Medical Image Encryption Using Single-Round DNA-Chaos

OVERVIEW:

This scheme proposes lightweight medical image encryption using single-round chaos-based diffusion with dynamic Deoxyribonucleic Acid (DNA) encoding operations. With the rapid proliferation of medical imaging systems and cloud-based healthcare services, protecting sensitive image data has become crucial. Traditional encryption algorithms like AES and RSA, while secure, are computationally intensive and not optimized for image data. This scheme handles and preserves multiple intensity levels in 8-bit medical images, and maintains perfect reconstruction capability with zero information loss. This paper proposes an encryption scheme that employs single-phase permutation-diffusion-DNA processing that reduces the complexity of computation but maintains cryptographic strength. The approach combines with SHA-256, logistic chaotic map, DNA encoding using three arithmetic operations (addition, subtraction, and XOR). The system uses a password that is combined with the image to generate SHA-256 seeds that dynamically determine chaotic map parameters, DNA encoding rules, and operations. This dynamic approach ensures that different images yield diverse encryption patterns, eliminating predictable sequences and enhancing security. The security evaluation shows strong performance across multiple metrics such as high entropy, ideal NPCR and UACI values, and the enhanced key space provides robust protection against brute-force and statistical attacks. The low complexity of this approach makes it highly suitable for fast and secure medical image encryption in practical healthcare systems.

DATASET: Any Medical image (8 and 16 bit images)-grayscale

WORKFLOW DIAGRAM:

<img width="1920" height="1080" alt="Flow Chart" src="https://github.com/user-attachments/assets/254f3ed5-188f-48de-8a6a-a0fcdeb41d11" />
