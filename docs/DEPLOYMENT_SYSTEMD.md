# Deployment with Systemd

## 1. Setup Service File

Copy the example service file to systemd:

```bash
sudo cp deploy/systemd/freqtrade-icicibreeze.service.example /etc/systemd/system/freqtrade-icicibreeze.service
```

## 2. Edit Configuration

Edit the service file to match your paths and user:

```bash
sudo nano /etc/systemd/system/freqtrade-icicibreeze.service
```

- Update `User=`, `WorkingDirectory=`, and `ExecStart=` paths.
- Set environment variables in the service or link an `.env` file.

## 3. Enable and Start

```bash
sudo systemctl daemon-reload
sudo systemctl enable freqtrade-icicibreeze
sudo systemctl start freqtrade-icicibreeze
```

## 4. Logs

View logs with journalctl:

```bash
sudo journalctl -u freqtrade-icicibreeze -f
```
