// Android background task (Capacitor Background Runner), runs about every 30 minutes even when
// LioScan is closed: looks for a newer published version and shows a notification once per version.
// The app tells this task its current version and where updates are published (event "setVersion").
addEventListener('setVersion', (resolve, reject, args) => {
  try {
    CapacitorKV.set('version', String(args.version))
    CapacitorKV.set('base', String(args.base))
    resolve()
  } catch (e) {
    reject(e)
  }
})

addEventListener('checkUpdate', async (resolve, reject) => {
  try {
    const base = CapacitorKV.get('base')
    const current = CapacitorKV.get('version')
    if (!base || !base.value || !current || !current.value) return resolve()
    const r = await fetch(base.value + 'latest.json?t=' + Date.now())
    const latest = await r.json()
    const version = JSON.parse(latest.payload).version
    const notified = CapacitorKV.get('notified')
    if (version > current.value && (!notified || notified.value !== version)) {
      CapacitorNotifications.schedule([
        {
          id: 4242,
          title: 'Mise à jour de LioScan disponible',
          body: 'La version ' + version + ' est prête : touche pour l’installer et voir les nouveautés.',
          scheduleAt: new Date(Date.now() + 1000),
        },
      ])
      CapacitorKV.set('notified', version)
    }
    resolve()
  } catch (e) {
    reject(e)
  }
})
