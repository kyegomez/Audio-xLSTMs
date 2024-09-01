[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

# Audio xLSTMs: Learning Self-supervised audio representations with xLSTMs

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

This is a community based approach to an implementation mostly for practice. I will implement the model architecture as defined in the paper but will leave someone else to implement the training script! So please create a training script if you have the time and energy


# Install
```bash
$ pip3 install -U audio-xlstm
```

# License
MIT

# Todo

- [ ] Implement the flip module
- [ ] Correctly leverage msltm module
- [ ] Ensure model architecture is correct
- [ ] Implement training script on whisper like data
- [ ] Implement speech and audio recognition datasets


# Citation
```bibtex
@article{xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}

```