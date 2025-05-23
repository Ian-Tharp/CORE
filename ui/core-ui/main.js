console.log('Electron main process started');

const { app, BrowserWindow } = require('electron');
const path = require('path');
const url = require('url');

let win;

const isDev = process.env.NODE_ENV === 'development' || process.env['ELECTRON_IS_DEV'] === '1';

function createWindow() {
  console.log('Creating browser window...');
  win = new BrowserWindow({
    width: 1920,
    height: 1080,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false, // Be cautious with this in production, consider alternatives
    },
  });

  if (isDev) {
    console.log('Loading http://localhost:4200');
    win.loadURL('http://localhost:4200');
  } else {
    const indexPath = path.join(__dirname, 'dist/core-ui/browser/index.html');
    console.log('Loading file:', indexPath);
    win.loadURL(
      url.format({
        pathname: indexPath,
        protocol: 'file:',
        slashes: true,
      })
    );
  }

  // The following is optional and will open the DevTools:
  // win.webContents.openDevTools()

  win.on('closed', () => {
    win = null;
  });

  win.on('unresponsive', () => {
    console.error('Window is unresponsive!');
  });

  win.on('crashed', () => {
    console.error('Window has crashed!');
  });
}

app.on('ready', () => {
  console.log('App is ready');
  createWindow();
});

// on macOS, closing the window doesn't quit the app
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// If your app has no windows, close the app when called
app.on('activate', () => {
  if (win === null) {
    createWindow();
  }
});

app.on('will-quit', () => {
  console.log('App will quit');
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
}); 